import paddle
import paddle.nn as nn
import pgl
from utils import AdoptiveSoftThreshold
seed = 10
paddle.seed(seed)

class GCN(nn.Layer):
    """Implement of GCN
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=2,
                 hidden_size=256,
                 **kwargs):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
        self.output = nn.Linear(self.hidden_size, self.num_class, weight_attr=nn.initializer.KaimingUniform())
    def forward(self, graph, feature):
        for m in self.gcns:
            feature = m(graph, feature)
        logits = self.output(feature)
        return logits

class Attention(nn.Layer):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, z):
        w = self.project(z)
        beta = paddle.nn.functional.softmax(w, axis=1)
        return (beta * z).sum(1), beta

class SEGN(nn.Layer):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(SEGN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        # layers
        self.special_emb_exp = GCN(input_size=self.in_dim, num_class=self.out_dim, hidden_size=self.hid_dim)
        self.special_emb_methy = GCN(input_size=self.in_dim, num_class=self.out_dim, hidden_size=self.hid_dim)
        self.special_emb_mirna = GCN(input_size=self.in_dim, num_class=self.out_dim, hidden_size=self.hid_dim)

        self.common_emb = GCN(input_size=self.in_dim, num_class=self.out_dim, hidden_size=self.hid_dim)
        self.threshold = AdoptiveSoftThreshold(1)

        self.hz_MLP = nn.Linear(self.out_dim, self.out_dim, weight_attr=nn.initializer.KaimingUniform())

        self.attention = Attention(out_dim, hidden_size=hid_dim)
        # hyparameters
        self.shrink = 1.0 / out_dim

    def special_embedding(self, graphs, features):
        emb_1 = self.special_emb_exp(graphs[0], features[0])
        emb_2 = self.special_emb_methy(graphs[1], features[1])
        emb_3 = self.special_emb_mirna(graphs[2], features[2])

        return paddle.stack([emb_1, emb_2, emb_3], axis=1)

    def shared_embedding(self, graphs, features):
        shared_emb = self.common_emb(graphs, features)
        return shared_emb

    def get_coef(self, query, keys):
        c = self.threshold(paddle.mm(query, keys.T))
        return self.shrink * c

    def forward(self, graphs, features):
        shared_1 = self.shared_embedding(graphs[0], features[0])
        shared_2 = self.shared_embedding(graphs[1], features[1])
        shared_3 = self.shared_embedding(graphs[2], features[2])
        shared = (shared_1 + shared_2 + shared_3) / 3

        special = self.special_embedding(graphs, features)
        emb, att = self.attention(special)
        out = self.get_coef(shared, emb)
        h_persudo = self.hz_MLP(shared)

        return out, shared, emb, att, h_persudo
