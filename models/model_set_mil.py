from models.attention_modules import *
from torch.nn import init
from models.gcn_layer import GraphConvolution


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant_(m.bias.data, 0.0)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=None, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout is not None:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, n_classes=4, affinity_threshold=None):
        super(MIL_Attention_FC_surv, self).__init__()

        self.alpha = 0.4
        self.dropout_rate = 0.25

        self.affinity_threshold = affinity_threshold
        self.dimension_0, self.dimension_1 = 512, 512
        self.dropout = nn.Dropout(self.dropout_rate)

        self.non_lin0 = nn.Sequential(nn.Linear(self.dimension_0 * 2, self.dimension_0),
                                     nn.LayerNorm(self.dimension_0),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.dimension_0, self.dimension_0))
        
        self.non_lin1 = nn.Sequential(nn.Linear(self.dimension_0 * 2, self.dimension_0),
                                     nn.LayerNorm(self.dimension_0),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.dimension_0, self.dimension_0))

        self.gcn_cos_0 = GraphConvolution(self.dimension_0, self.dimension_1)
        self.gcn_cos_norm_0 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.gcn_cos_1 = GraphConvolution(self.dimension_1, self.dimension_1)
        self.gcn_cos_norm_1 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.gcn_cos_2 = GraphConvolution(self.dimension_1, self.dimension_1)
        self.gcn_cos_norm_2 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.gcn_auto_0 = GraphConvolution(self.dimension_0, self.dimension_1)
        self.gcn_auto_norm_0 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.gcn_auto_1 = GraphConvolution(self.dimension_1, self.dimension_1)
        self.gcn_auto_norm_1 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.gcn_auto_2 = GraphConvolution(self.dimension_1, self.dimension_1)
        self.gcn_auto_norm_2 = nn.Sequential(
            nn.LayerNorm(self.dimension_1),
            nn.ReLU(inplace=True)
        )

        self.attn = Attn_Net_Gated(dropout=self.dropout_rate)
 
        # self.mlp = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     # nn.LayerNorm(512),
        #     # nn.ReLU(inplace=True),
        #     nn.Dropout(self.dropout_rate)
        # )
        
        self.classifier_global = nn.Linear(self.dimension_1*2, n_classes, bias=False)
        self.classifier_global.apply(weights_init_classifier)
        
        self.ln = nn.LayerNorm(1024)

    def forward_global(self, x):
        logits = self.classifier_global(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat


    def cosine_sim_cal(self, x):
        x = F.normalize(x, dim=1)
        cos_sim = torch.mm(x, x.t())
        cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())
        cos_sim = (cos_sim >= self.affinity_threshold).type(torch.float32)
        return cos_sim

    def attn_sim_cal(self, x):
        attention_weights = self.attn(x)

        agg_weights = attention_weights

        attention_weights = torch.sigmoid(attention_weights)

        attention_weights = attention_weights * attention_weights.t()
        attn_min, attn_max = attention_weights.min(), attention_weights.max()
        attention_weights = (attention_weights - attn_min)/(attn_max - attn_min)
        return attention_weights, agg_weights

    def forward(self, epoch=None, **kwargs):
        x_path = kwargs['x_path']
        maps = kwargs['maps']


        cos_sim = self.cosine_sim_cal(x_path)
        attention_weights, agg_weights = self.attn_sim_cal(x_path)
        attention_weights = self.alpha + (1 - self.alpha) * attention_weights + maps

        # GCN version 
        # with torch.no_grad():    
        #     D_cos = torch.diag(cos_sim.sum(0)**(-0.5)) 
        #     cos_sim = D_cos.mm(cos_sim).mm(D_cos)               
        # D_atten = torch.diag(attention_weights.sum(0)**(-0.5)) 
        # attention_weights = D_atten.mm(attention_weights).mm(D_atten)                               

        # Simplified strategy with weights L1 Norm
        cos_sim = F.normalize(cos_sim, p=1)
        attention_weights = F.normalize(attention_weights, p=1)
        
        # specifically designed for 10 epoch.
        if epoch is not None and epoch <= 10:
            self.alpha = 0.3 - epoch // 5 * 0.1

        x0 = self.non_lin0(x_path)
        x1 = self.non_lin1(x_path)

        gc1 = self.gcn_cos_norm_0(self.gcn_cos_0(x0, cos_sim))
        gc1 = self.gcn_cos_norm_1(self.gcn_cos_1(gc1, cos_sim))
        gc1 = self.gcn_cos_norm_2(self.gcn_cos_2(gc1, cos_sim))

        gc2 = self.gcn_auto_norm_0(self.gcn_auto_0(x1, attention_weights))
        gc2 = self.gcn_auto_norm_1(self.gcn_auto_1(gc2, attention_weights))
        gc2 = self.gcn_auto_norm_2(self.gcn_auto_2(gc2, attention_weights))

        gc_final = torch.cat((gc2, gc1), dim=1)

        length = agg_weights.size(0)
        trans_weights = torch.matmul(cos_sim, agg_weights) / length**0.5
        final_weights = 0.3 * trans_weights + 0.7 * agg_weights

        final_weights = F.softmax(final_weights, dim=0)
        gc_final = torch.matmul(final_weights.t(), gc_final)

        gc_final = self.ln(gc_final)

        hazards_gl, S_gl, Y_hat_gl = self.forward_global(gc_final)

        return hazards_gl, S_gl, Y_hat_gl

