import torch
import torch.nn as nn
import copy
import math
from torch.nn import functional as F
from copy import deepcopy
from timm.models.layers import trunc_normal_
from models.vit import VisionTransformer, PatchEmbed, Block, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        return out


class ViT_Prompts(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        init_values=None,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
    ):

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            init_values=init_values,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )

    def forward(self, x, instance_tokens=None, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )

        x = x + self.pos_embed.to(x.dtype)
        if instance_tokens is not None:
            x = torch.cat([x[:, :1, :], instance_tokens, x[:, 1:, :]], dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        return x




def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None))
    model = build_model_with_cfg(
        ViT_Prompts,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs
    )
    return model


class DceNet(nn.Module):

    def __init__(self, args):
        super(DceNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        if args["21k"] == 1:
            self.image_encoder = _create_vision_transformer(
                "vit_base_patch16_224_in21k", pretrained=True, **model_kwargs)
            print("Using 21k model")
        else:
            self.image_encoder = _create_vision_transformer(
                "vit_base_patch16_224", pretrained=True, **model_kwargs)
        self.use_sm = 0
        self.temp = args["temp"]
        self.bal_epoch = args["bal_epoch"]
        self.dataset = args["dataset"]
        self.class_num = 1
        if args["dataset"] == "cddb":
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.class_num = 50
        elif args["dataset"] == "officehome":
            self.class_num = 65
        else:
            raise ValueError("Unknown datasets: {}.".format(args["dataset"]))
        self.prompt_type = args["prompt_type"]
        self.classifier_pool_naive = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args["embd_dim"], args["embd_dim"]),
                    nn.ReLU(),
                    nn.Linear(args["embd_dim"], args["embd_dim"] // 2),
                    CosineLinear(args["embd_dim"] // 2, self.class_num),
                )
                for i in range(args["total_sessions"])
            ]
        )
        self.classifier_pool_bal = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args["embd_dim"], args["embd_dim"]),
                    nn.ReLU(),
                    nn.Linear(args["embd_dim"], args["embd_dim"] // 2),
                    CosineLinear(args["embd_dim"] // 2, self.class_num),
                )
                for i in range(args["total_sessions"])
            ]
        )
        self.classifier_pool_rev = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args["embd_dim"], args["embd_dim"]),
                    nn.ReLU(),
                    nn.Linear(args["embd_dim"], args["embd_dim"] // 2),
                    CosineLinear(args["embd_dim"] // 2, self.class_num),
                )
                for i in range(args["total_sessions"])
            ]
        )
        if self.prompt_type == "one" or self.prompt_type == "all":
            self.prompt_pool = nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
        elif self.prompt_type == "no":
            self.prompt_pool = None

        self.numtask = 0

    def init_select_network(self, task_id):
        select_dim = task_id * 3
        self.select_network = nn.Sequential(
            nn.Linear(self.image_encoder.embed_dim, self.image_encoder.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.image_encoder.embed_dim // 2, select_dim),
        )

    def get_domain_param_list(self):
        return [p for p in self.select_network.parameters()]

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        if self.prompt_pool is not None:
            image_features = self.image_encoder(image, self.prompt_pool.weight)
        else:
            image_features = self.image_encoder(image)
        return image_features

    def forward(self, image, train=False, weight=None, quick=False):
        if train:
            logits = []
            if self.prompt_pool is not None:
                image_features = self.image_encoder(image, self.prompt_pool.weight)
            else:
                with torch.no_grad():
                    image_features = self.image_encoder(image)
            logits = self.classifier_pool_naive[self.numtask - 1](image_features)
            bal_logits = self.classifier_pool_bal[self.numtask - 1](image_features)
            rev_logits = self.classifier_pool_rev[self.numtask - 1](image_features)

            return {
                "logits": logits,
                "bal_logits": bal_logits,
                "rev_logits": rev_logits,
            }
        else:
            if self.prompt_pool is not None:
                image_features = self.image_encoder(image, self.prompt_pool.weight)
            else:
                image_features = self.image_encoder(image)
            if quick:
                return image_features
            weight_logits, all_logits, last_logits, all_logits_bal, data_weight = self.forward_head(
                image_features)
            return {
                "logits": weight_logits,
                "all_logits": all_logits,
                "last_logits": last_logits,
                "all_logits_bal": all_logits_bal,
                "data_weight": data_weight,
            }

    def forward_head(self, img_features):
        all_logits = []
        all_logits_bal = []
        for p_id in range(self.numtask):
            all_logits.append(self.classifier_pool_naive[p_id](img_features).detach())
            all_logits.append(self.classifier_pool_bal[p_id](img_features).detach())
            all_logits_bal.append(all_logits[-1])
            if p_id == self.numtask - 1:
                last_logits = all_logits[-1]
            all_logits.append(self.classifier_pool_rev[p_id](img_features).detach())
        stack_logits = torch.stack(all_logits, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        all_logits_bal = torch.cat(all_logits_bal, dim=1)
        logits_weight = self.select_network(img_features)
        if self.use_sm == 1:
            sm_weight = torch.nn.functional.softmax(logits_weight * self.temp, dim=1)
            sm_weight = sm_weight.unsqueeze(-1)
            weight_logits = torch.sum(stack_logits * sm_weight, dim=1)
        else:
            logits_weight = logits_weight.unsqueeze(-1)
            weight_logits = torch.sum(stack_logits * logits_weight, dim=1)
        return weight_logits, all_logits, all_logits_bal, last_logits, logits_weight

    def forward_head_q(self, img_features):
        all_logits = []
        for p_id in range(self.numtask):
            all_logits.append(self.classifier_pool_naive[p_id](img_features).detach())
            all_logits.append(self.classifier_pool_bal[p_id](img_features).detach())
            all_logits.append(self.classifier_pool_rev[p_id](img_features).detach())
        stack_logits = torch.stack(all_logits, dim=1)
        logits_weight = self.select_network(img_features)
        logits_weight = logits_weight.unsqueeze(-1)
        weight_logits = torch.sum(stack_logits * logits_weight, dim=1)
        return weight_logits

    def update_fc(self, nb_classes):
        self.numtask += 1
        self.init_select_network(self.numtask)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
