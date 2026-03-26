from functools import partial  # 添加这一行，导入partial函数
import torch
import torch.nn as nn
# from utils.pos_embed import get_1d_sincos_pos_embed_from_grid


from models.utils.pos_embed import get_1d_sincos_pos_embed_from_grid


# from models.utils.pos_embed import get_1d_sincos_pos_embed

class MaskedAutoencoderViT1D(nn.Module):
    def __init__(self, patch_size=18, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 seq_length=1800, norm_pix_loss=True):
        super(MaskedAutoencoderViT1D, self).__init__()

        self.patch_size = patch_size
        self.seq_length = seq_length
        # 计算 num_patches
        self.num_patches = seq_length // patch_size  # 输入长度 1800 / patch_size = num_patches

        #
        self.patch_embed = self.PatchEmbed1D(patch_size=patch_size, embed_dim=embed_dim)

        # 使用正弦余弦位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        # 定义 cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # cls_token 用于编码输入的信息

        # Transformer 编码器部分
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 解码器部分
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # 创建适应的 decoder_pos_embed
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # 位置编码的初始化
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.num_patches + 1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], self.num_patches + 1)

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # 初始化embedding参数
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 初始化分类头和masked token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        将输入的一维序列 (N, seq_length) 转换为多个 patch (N, num_patches, patch_size)
        x: [batch_size, seq_length]

        返回:
        x_patches: [batch_size, num_patches, patch_size]
        """
        batch_size, seq_length = x.shape
        patch_size = self.patch_size

        # 确保 seq_length 可以被 patch_size 整除
        assert seq_length % patch_size == 0, "Sequence length must be divisible by patch size"

        # 计算 num_patches
        num_patches = seq_length // patch_size

        # 重塑输入，分割为多个 patch
        x_patches = x.view(batch_size, num_patches, patch_size)

        return x_patches

    def unpatchify(self, x):
        """
        将经过 patchify 后的序列还原为原始的一维序列 (N, seq_length)

        x: [batch_size, num_patches, patch_size]  --> 恢复成 [batch_size, seq_length]

        返回:
        x: [batch_size, seq_length]
        """
        batch_size, num_patches, patch_size = x.shape
        seq_length = num_patches * patch_size  # 恢复回原始的 seq_length

        # 将 patches 拼接回原始的一维序列
        x = x.view(batch_size, seq_length)

        return x

    def forward_encoder(self, x, mask_ratio):
        # 编码器部分
        x = self.patch_embed(x)  # [N, num_patches, embed_dim]
        x = x + self.pos_embed[:, 1:, :]  # 加上位置编码，但不包括 cls_token 的位置编码

        # 随机遮盖
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Transformer 编码,加上分类头
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 处理 cls_token 的位置编码
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 解码器部分
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # 添加解码头

        x = x + self.decoder_pos_embed

        # 解码器 Transformer
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_loss(self, x, pred, mask):
        """
        x: [N, seq_length] (输入的一维序列)
        pred: [N, num_patches, patch_size] (模型预测的重建 patch)
        mask: [N, num_patches] (掩码，0 表示保留，1 表示去除，计算损失时需要关注被去除的部分)
        """
        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

        # 动态权重（例如，基于掩码区域的难度）
        # dynamic_weights = mask * 2  # 根据掩码区域调整权重，举例：掩码区域权重为 2，非掩码区域为 1
        # weighted_loss = loss * dynamic_weights
        #
        # # 计算损失时只关注去除的部分
        # weighted_loss = (weighted_loss * mask).sum() / mask.sum()  # 只计算去掉部分的损失
        #
        # # 假设 pred 形状为 (N, L, C) 或 (N, C, L)，只要 dim=-1 是你要差分的维度即可
        #
        # # --- 一阶导数修正 ---
        # # append 的张量在 dim=-1 上的长度为 1，其他维度与 pred 保持一致
        # append_1st = torch.zeros((*pred.shape[:-1], 1), device=pred.device)
        # pred_diff_1st = torch.diff(pred, dim=-1, append=append_1st)
        # target_diff_1st = torch.diff(target, dim=-1, append=append_1st)
        #
        # # --- 二阶导数修正 ---
        # # 注意：二阶导数是对一阶导数的结果再求导。
        # # 为了保持尺寸一致，append 的长度依然建议设为 1（因为 diff 每次减少 1 个元素）
        # append_2nd = torch.zeros((*pred_diff_1st.shape[:-1], 1), device=pred.device)
        # pred_diff_2nd = torch.diff(pred_diff_1st, dim=-1, append=append_2nd)
        # target_diff_2nd = torch.diff(target_diff_1st, dim=-1, append=append_2nd)
        #
        # # 计算损失
        # grad_loss = torch.mean((pred_diff_1st - target_diff_1st) ** 2)
        # grad_loss += torch.mean((pred_diff_2nd - target_diff_2nd) ** 2)
        #
        # # 组合L1损失和梯度损失
        # total_loss = weighted_loss + grad_loss

    def predeict(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # loss = self.forward_loss(x, pred, mask)
        out = self.unpatchify(pred)
        return out

    class PatchEmbed1D(nn.Module):
        def __init__(self, patch_size, embed_dim):
            super().__init__()
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.proj = nn.Linear(patch_size, embed_dim)

        def forward(self, x):
            # x: [batch_size, seq_length]
            batch_size, seq_length = x.shape

            # 确保 seq_length 被 patch_size 整除
            num_patches = seq_length // self.patch_size
            assert seq_length % self.patch_size == 0, "Sequence length must be divisible by patch size"

            # 重塑为 [batch_size, num_patches, patch_size]
            x = x.view(batch_size, num_patches, self.patch_size)

            # 通过线性变换映射到 embedding
            x = self.proj(x)  # [batch_size, num_patches, embed_dim]

            return x


def Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, norm_layer):
    return nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=0.1,
        activation='gelu',
        batch_first=True  # ❗关键修复
    )


# 用于创建模型的辅助函数
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(
        patch_size=9, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT1D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    model = mae_vit_base_patch16_dec512d8b()
    # 设置随机种子以确保结果可重现
    torch.manual_seed(0)

    # 假设输入的序列长度是1800，batch_size=4
    batch_size = 4
    seq_length = 1800
    # patch_size = 9  # 每个patch的大小

    # 创建一个输入的随机一维序列 (batch_size, seq_length)
    x = torch.randint(0, 256, (batch_size, seq_length), dtype=torch.float32)

    # 打印输入的形状，确保它是 [batch_size, seq_length]
    print(f"Input shape: {x.shape}")

    # 使用 patchify 将输入切分为多个 patch
    x_patches = model.patchify(x)
    print(f"Patches shape: {x_patches.shape}")  # [batch_size, num_patches, patch_size]

    # 使用 unpatchify 将 patch 还原为原始输入
    x_reconstructed = model.unpatchify(x_patches)
    print(f"Reconstructed shape: {x_reconstructed.shape}")  # [batch_size, seq_length]

    # 验证是否恢复正确：原始输入和还原后的输入应该完全相同
    assert torch.allclose(x, x_reconstructed), "The original input and reconstructed input do not match!"
    print("Test passed! Patchify and Unpatchify work correctly.")

    y = model(x)
