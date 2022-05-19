import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from coatnet import CoAtNet
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torchvision
import timm
from copy import deepcopy
from utils import apk

def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device)


class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class ImageModel(nn.Module):

    def __init__(self,img_model, dropout=0.5):
        super().__init__()
        if img_model == "efficient":
            self.eff = EfficientNet.from_pretrained('efficientnet-b7')
            # self.eff._fc = nn.Linear(2048, 128)
            self.eff._fc = nn.Linear(self.eff._fc.in_features,128)
        elif img_model == "ViT":
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True,num_classes=128)
        elif img_model == "ns_efficient":
            self.ns_efficient = timm.create_model("tf_efficientnet_b0_ns",pretrained=True,num_classes=128)
        if img_model == "inception":
            self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            self.inception.avgpool = nn.AvgPool2d((8, 8))
            self.inception.dropout = nn.Dropout(dropout)
            self.inception.fc = nn.Linear(2048, 128)
        elif img_model == "resnet":
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.resnet.fc = nn.Linear(512,128)
        elif img_model == "coatnet":
            num_blocks = [2, 2, 6, 14, 2]  # L
            channels = [64, 96, 192, 384, 768]  # D
            self.coatnet =  CoAtNet((224, 224), 3, num_blocks, channels, num_classes=128)
        elif img_model == "resnet152":
            self.resnet = torchvision.models.resnet152(pretrained=True)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features,128)
        elif img_model == "resnet_v2":
            self.res = timm.create_model("resnetv2_101x1_bitm",pretrained=True,num_classes=128)
        self.img_model = img_model
    def forward(self, img):
        if self.img_model == "inception":
            out = self.inception(img)
        elif self.img_model == "resnet":
            out = self.resnet(img)
        elif self.img_model == "coatnet":
            out = self.coatnet(img)
        elif self.img_model == "efficient":
            out = self.eff(img)
        elif self.img_model == "resnet152":
            out = self.fc(self.resnet(img))
        elif self.img_model == "resnet_v2":
            out = self.res(img)
        elif self.img_model == "ViT":
            out = self.vit(img)
        elif self.img_model == "ns_efficient":
            out = self.ns_efficient(img)
        return out

class TextModel(nn.Module):

    def __init__(self, pretrained_langaue_model,old_checkpoint=False):
        super().__init__()
        self.pretrained_language_model = pretrained_langaue_model
        self.old_checkpoint = old_checkpoint
        if old_checkpoint:
            self.bert = AutoModel.from_pretrained(pretrained_langaue_model)
        else:
            self.encoder = AutoModel.from_pretrained(pretrained_langaue_model)
        self.config = AutoConfig.from_pretrained(pretrained_langaue_model)
        self.lstm = nn.LSTM(input_size=(self.config.hidden_size), hidden_size=128, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None):
        max_seq_len = input_ids.size()[1]
        non_pad_tensor = input_ids != 0
        seq_lengths = non_pad_tensor.sum(-1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        input_ids = input_ids[perm_idx]
        attention_mask = attention_mask[perm_idx]
        if self.old_checkpoint:
            outputs = self.bert(input_ids, attention_mask)['last_hidden_state']
        else:
            outputs = self.encoder(input_ids, attention_mask)['last_hidden_state']
        packed_input = pack_padded_sequence(outputs, (seq_lengths.detach().cpu().numpy()), batch_first=True)
        packed_output, (_, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        masks = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(output.size(0), max_seq_len, output.size(2))
        output = output.gather(1, masks)[:, 0, :]
        sorted_perm_idx, idx = perm_idx.sort(0, descending=False)
        output = output[idx]
        return output


class FusionLayer(nn.Module):

    def __init__(self, pretrained_langaue_model,img_model,metric_learning=False,distributed=None,low_rank_fusion=False,use_triplet=False,dropout=0.3,pos_weight=None,parallel=True,old_checkpoint = False):
        super().__init__()
        self.old_checkpoint = old_checkpoint
        self.text_model = TextModel(pretrained_langaue_model,old_checkpoint=old_checkpoint)
        self.img_model = ImageModel(img_model)
        self.img_model_name = img_model
        self.fc1 = nn.Linear(256, 256)
        self.metric_learning = metric_learning
        self.parallel = parallel
        self.distributed = distributed
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        if pos_weight is not None:
            if not self.parallel or self.distributed:
                self.criterion = nn.BCEWithLogitsLoss(reduction='sum',pos_weight=torch.tensor(pos_weight))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.low_rank_fustion = low_rank_fusion
        self.use_triplet = use_triplet
        self.l2 = L2_norm()
        self.dropout = nn.Dropout(dropout)
        if self.low_rank_fustion:
            self.rank = 5
            self.post_fusion_prob = 0.3
            self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
            # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
            self.img_factor = Parameter(torch.Tensor(self.rank, 128 + 1,1))
            self.text_factor = Parameter(torch.Tensor(self.rank, 128 + 1,1))
            self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
            self.fusion_bias = Parameter(torch.Tensor(1, 1))
            self.output_dim = 1
            # init teh factors
            xavier_normal_(self.img_factor)
            xavier_normal_(self.text_factor)
            xavier_normal_(self.fusion_weights)
            self.fusion_bias.data.fill_(0)

    def forward(self,id=None, img_tensor=None, input_ids=None, attention_mask=None, label=None):
        if label is not None:
            if self.img_model_name == "coatnet":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "resnet":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "efficient":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "resnet152":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "resnet_v2":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "ViT":
                out_img = self.img_model(img_tensor)
            elif self.img_model_name == "ns_efficient":
                out_img = self.img_model(img_tensor)
            else:
                if self.training:
                    out_img = self.img_model(img_tensor)[0]
                else:
                    out_img = self.img_model(img_tensor)
        else:
            if self.img_model == "inception":
                out_img = self.img_model(img_tensor)[0]
            else:
                out_img = self.img_model(img_tensor)
        out_text = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # out_text = self.bn(out_text)
        # out_img = self.bn(out_img)
        if self.low_rank_fustion:
            if out_img.is_cuda:
                DTYPE = torch.cuda.FloatTensor
            else:
                DTYPE = torch.FloatTensor
            batch_size = out_img.data.shape[0]
            _img_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), out_img), dim=1)
            _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), out_text), dim=1)
            fusion_img = torch.matmul(_img_h, self.img_factor)
            fusion_text = torch.matmul(_text_h, self.text_factor)
            fusion_zy = fusion_img * fusion_text

            # output = torch.sum(fusion_zy, dim=0).squeeze()
            # use linear transformation instead of simple summation, more flexibility
            output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
            output = output.view(-1, self.output_dim)
            if label is not None:
                 loss = self.criterion(output, label)
                 return (loss, torch.sigmoid(output.squeeze(-1)))
            return output

        else:
            logits = torch.cat((out_img, out_text), 1)
            out1 = self.fc1(logits)
            out = self.fc2(self.relu(out1))
            if label is not None:
                 if self.parallel or self.distributed:
                    if self.training:
                        return out,label
                    return out, label, torch.sigmoid(out.squeeze(-1))

                 if self.use_triplet:
                     triploss = TripletLoss(out.device)
                     norm_out = self.l2(out)
                     loss = self.criterion(out, label) + triploss(norm_out,label.squeeze(-1))
                     return (loss, torch.sigmoid(out.squeeze(-1)))
                 else:
                     loss = self.criterion(out, label)
                     return (loss, torch.sigmoid(out.squeeze(-1)))
            return out
        return out1


if __name__ == '__main__':
    pass