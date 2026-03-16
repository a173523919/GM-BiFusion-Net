import torch
import torch.nn as nn
import torch.nn.functional as F
from .csms6s import selective_scan_fn
import math
from functools import partial

# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        """
        dt_rank：动态时间投影的秩，即输入维度。
        d_inner：隐藏维度，即输出维度。
        dt_scale：权重初始化的比例因子，默认值为 1.0。
        dt_init：权重初始化的方式，可以是 "constant" 或 "random"。
        dt_min：时间步长的最小值，默认为 0.001。
        dt_max：时间步长的最大值，默认为 0.1。
        dt_init_floor：时间步长初始化的下限，默认为 1e-4
        """

        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        #如果是 "constant"，将权重初始化为 dt_init_std。
        #如果是 "random"，将权重初始化为均匀分布 
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        #计算 Δt 的逆 Softplus 值
        dt = torch.exp(
            #随机生成一个时间步长 Δt，使其在范围 [dt_min,dt_max] 内。
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        #将计算得到的 inv_dt 赋值给全连接层的偏置项,dt_proj。
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    """
    self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
    )"""
    #装饰器，用于定义类方法。
    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        #cls表示类本身，调用类本身的dt_init方法
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        #将dt_projs中的每一层的参数取出，再按照dim=0转化为可训练的torch参数，得到可训练层
        #这样做的目的是实现分组并行训练
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================AD的初始化
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SS2Dv0(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        #输入特征的维度。
        d_model=96,
        #状态空间的维度
        d_state=16,
        #隐藏维度与输入维度的比例。
        ssm_ratio=2.0,
        #动态时间投影的秩。dt_rank 表示时间控制变量 dt 对应的可训练参数矩阵的大小。
        dt_rank="auto",
        # ======================
        #dropout 的概率。
        dropout=0.0,
        # ======================
        #是否按序列方法处理数据
        seq=False,
        #是否强制使用 FP32 精度
        force_fp32=True,
        **kwargs,
    ):
        #确保“kwargs = {"channel_first": False}”
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]


        #定义默认参数
        #时间步长（dt）的最小值。时间步长用于控制状态更新的速度。
        dt_min = 0.001
        #时间步长（dt）的最大值
        dt_max = 0.1
        #时间步长的初始化方式。
        dt_init = "random"
        #用于调整时间步长的初始值范围。例如，dt_scale = 1.0 表示时间步长的初始值范围为 [dt_min, dt_max]
        dt_scale = 1.0
        #时间步长初始化的下限。在初始化时，确保时间步长不会低于这个值，避免数值问题。
        dt_init_floor = 1e-4
        #是否在全连接层（nn.Linear）中使用偏置项。
        bias = False
        #是否在卷积层（nn.Conv2d）中使用偏置项。
        conv_bias = True
        #卷积核的大小
        d_conv = 3
        #分组数量
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}

        #调用构造函数
        super().__init__()
        #计算隐藏层的维度
        d_inner = int(ssm_ratio * d_model)
        #确定动态时间投影的秩 math.ceil 是向上取整函数
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        #调用前向传播函数
        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        #线性投影层，处理输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        
        #定义silu层
        act_layer = nn.SiLU
        self.act: nn.Module = act_layer()

        #构建多个线性层
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        #遍历 self.x_proj 中的每个全连接层，提取它们的权重矩阵 t.weight
        #将堆叠后的权重张量转换为一个 PyTorch 参数（nn.Parameter），这样它就可以被自动识别为模型的可训练参数，并参与梯度计算和优化。
        #将x_proj中的每一层的参数取出，再按照dim=0转化为可训练的torch参数，得到一个(K, N, inner)大小的可训练层
        #这样做的目的是实现分组并行训练
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        #   - K 是分组数量（k_group）
        #   - N 是每个分组的输出维度（dt_rank + d_state * 2）
        #   - inner 是每个分组的输入维度（d_inner）
        # nn.Linear 层的权重矩阵的形状总是 (out_features, in_features)
        #删除x_proj
        del self.x_proj

        # dt proj, A, D ============================
        #A_logs：在状态空间模型中，状态转移矩阵 A 描述了状态如何随时间变化。为了数值稳定性和可训练性，通常使用对数形式 log(A)。
        #self.Ds：在状态空间模型中，D 用于描述状态的直接跳过连接（skip connection），允许状态在时间步之间直接传递信息。
        #self.dt_projs_weight：在状态空间模型中，时间步长 Δt 通常是一个动态参数，用于控制状态更新的速度。self.dt_projs_weight 是用于计算 Δt 的权重矩阵。
        #self.dt_projs_bias：用于计算 Δt 的偏置向量。

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )
        # out proj =======================================
        #输出层和dropout
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        
        x = self.in_proj(x)
        
        #b（Batch Size）
        #h（Height）
        #w（Width）
        #d（Depth/Channels）

        #沿着最后一个维度分成两个部分
        #x.size=z.size=(b, h, w, d//2)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        
        #z经过一个silu，进行非原地操作
        #z = self.act(z.clone())
        #x 变形后经过一个卷积和silu
        x = x.permute(0, 3, 1, 2).contiguous()
        #x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        z = self.act(z)

        

        #创建一个新的函数 selective_scan，它是 selective_scan_fn 的一个特化版本。固定 selective_scan_fn 的 backend 参数为 "mamba"
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, D, H, W = x.shape
        """
        B：批量大小（Batch Size），表示一次处理的样本数量。
        D：特征维度（Depth/Channels），表示每个像素点的特征数量。
        H：高度（Height），表示图像的高度。
        W：宽度（Width），表示图像的宽度
        """
        D, N = self.A_logs.shape
        """
        D：特征维度（Depth/Channels），表示每个状态的特征数量。
        N：状态维度（State Dimension），表示状态的数量。
        """
        K, D, R = self.dt_projs_weight.shape
        """
        分组数量（Group Count），表示分组的数量。
        D：特征维度（Depth/Channels），表示每个分组的特征数量。
        R：时间投影的秩，在VSS中，使用一个嵌入序列表示时间偏移。
        """
        L = H * W

        """
        获得x_hwwh
        x.view(B, -1, L)：将 x 的空间维度（H 和 W）展平为一个长序列
        torch.transpose(x, dim0=2, dim1=3)：将 x 的高度和宽度维度进行转置，得到一个新的张量，其形状为 (B, D, W, H)
        .contiguous().view(B, -1, L) :先保证内存是连续的，再展平成一个长序列
        第一个张量：(B, D, L（HW）)，表示原始的水平方向特征。
        第二个张量：(B, D, L(WH))，表示转置后的垂直方向特征。
        操作：沿着新的维度 dim=1（D） 将这两个张量堆叠起来。
        .view(B, 2, -1, L)：输出一个二通道新张量
        """

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        
        #torch.flip(x_hwwh, dims=[-1])对x_hwwh的最一个维度进行反转，生成的张量 xs 的形状为 (B, 4, D, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        
        #这个 einsum 表达式的意思是，
        # 对于 xs 中的每个向量（大小为 d，长度为 l），
        #对于每个 l（序列长度），d 个序列与 c 个权重向量相乘后，会得到 c 个标量。这些标量会组合成一个新的张量，其形状为 (b, k, c, l)
        # 与 self.x_proj_weight 中的每个向量（大小为 d）进行点积，然后将结果按组 k 和输出维度 c 进行组合，最终得到一个形状为 b k c l 的张量

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        
        #b k c l
        #dts表示动态时间步长（Delta Time）
        #在状态空间模型中，Bs对应SSM的B
        #Cs 对应SSM的C
        #R代表，N代表状态 #变为可学习AD??
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        #使用dt_projs_weight再嵌入一次dts，使其与xs大小对应
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        #对xs的组和通道数进行压缩
        xs = xs.view(B, -1, L) # (b, k * d, l)
        #对dts进行同样操作
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        #B,C矩阵转换为(b, k, d_state=R（状态大小）, l)形式
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        #AD矩阵同理
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        #强制转换数据格式
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        #通过selective_scan处理xs,dts,As,Bs, Cs, Ds



        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = y.view(B, H, W, -1)
        y = y * z
        out = self.act(self.out_proj(y))
        return out

def example():
    # 创建一个随机输入张量
    # 假设输入数据的形状为 (batch_size, height, width, d_model)
    batch_size = 2
    height = 8
    width = 8
    d_model = 128
    # 创建随机输入数据
    input_data = torch.randn(batch_size, height, width, d_model)
    # 打印输入形状
    print("input_data shape:", input_data.shape)
    # 初始化 SS2Dv0 模型
    ss2d_model = SS2Dv0(
        d_model=d_model,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        seq=False,
        force_fp32=True
    )
    
    # 将模型和输入数据移至相同的设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss2d_model.to(device)
    input_data = input_data.to(device)
    
    # 前向传播
    output = ss2d_model(input_data)
    
    # 打印输出形状
    print("Output shape:", output.shape)

#example()


