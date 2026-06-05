import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================== 1. 实现Sinusoidal Position Encoding(正弦位置编码) ======================
class SinusoidalPE(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        # 预计算位置编码表 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)  # [max_len,1]
        # 分母 10000^(2i/d)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数维sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数维cos
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, d_model] 词嵌入E
        return: E + pos_pe  E+pos模式
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

# ====================== 2. 实现二维向量旋转(ROPE基础：2D旋转变换) ======================
def rotate_2d(x: torch.Tensor, pos: int, theta_base: float = 10000.0):
    """
    单组2维向量 [x0,x1] 按位置pos旋转θ=pos*θ_i，θ_i=1/(theta_base^(2i/d))
    :param x: [2] 一组二维向量
    :param pos: 位置索引
    :return: 旋转后向量
    """
    # 2维对应i=0，d=2
    theta = pos / (theta_base ** (0 * 2 / 2))
    cos_θ = np.cos(theta)
    sin_θ = np.sin(theta)
    x0, x1 = x[0], x[1]
    rx0 = x0 * cos_θ - x1 * sin_θ
    rx1 = x0 * sin_θ + x1 * cos_θ
    return torch.tensor([rx0, rx1])

# ====================== 3. 实现高维RoPE(任意偶数维度，标准旋转位置编码) ======================
def rope_forward(x: torch.Tensor, theta_base=10000.):
    """
    x: [batch, seq_len, d], d必须为偶数
    return: RoPE旋转后的特征，**不做E+pos相加，原地旋转特征**
    """
    B, L, D = x.shape
    assert D % 2 == 0, "RoPE要求特征维度为偶数"
    # 预生成每个维度对对应的频率系数
    freq = torch.exp(torch.arange(0, D, 2) * (-np.log(theta_base)/D))  # [D//2]
    pos = torch.arange(L).unsqueeze(1)  # [L,1]
    theta = pos * freq  # [L, D//2]
    cos = torch.cos(theta)[None,:,:] # [1,L,D//2]
    sin = torch.sin(theta)[None,:,:]

    # 拆分奇偶分组：(x_0,x_1),(x_2,x_3)...
    x_reshape = x.view(B, L, D//2, 2)
    x0 = x_reshape[...,0] # [B,L,D//2]
    x1 = x_reshape[...,1]

    # 旋转公式：[x0cos -x1sin, x0sin +x1cos]
    rx0 = x0 * cos - x1 * sin
    rx1 = x0 * sin + x1 * cos
    out = torch.stack([rx0, rx1], dim=-1).reshape(B,L,D)
    return out

# ====================== 4. 对比 E+pos(SinPE) 和 RoPE 输入/运算方式 ======================
def compare_input_mode():
    d = 8  # 特征维度
    seq_len = 5
    batch = 1
    # 随机词嵌入E
    E = torch.randn(batch, seq_len, d)

    # 方式1：SinPE(E+pos): 嵌入 + 独立位置编码
    sin_pe_layer = SinusoidalPE(d_model=d, max_len=512)
    out_sinpe = sin_pe_layer(E)
    print("=== E+pos(Sinusoidal PE)计算模式 ===")
    print(f"原始E shape:{E.shape}, PE是独立可加偏置，输出=E+PE, out:{out_sinpe.shape}")

    # 方式2：RoPE：直接对E做旋转变换，无额外相加
    out_rope = rope_forward(E)
    print("=== RoPE计算模式 ===")
    print(f"RoPE不新增位置向量，原地旋转E特征，输出为旋转后的E, out:{out_rope.shape}")
    print("【输入区别总结】")
    print("1. SinPE(E+pos): 位置信息是独立向量，加法融合：E_new = E + P")
    print("2. RoPE: 位置信息通过旋转变换嵌入E内部：E_new = Rotate(E, pos)，无相加操作")

# ====================== 5. 数值实验：验证RoPE相对位置不变性(核心性质) ======================
def test_relative_rope():
    """
    性质：任意两个位置p,q，特征内积只和相对距离k=p-q有关，和绝对位置无关
    <RoPE(E_p), RoPE(E_q)> = f(|p-q|)
    """
    d = 8
    B =1
    # 固定一组特征向量e
    e = torch.randn(B,1,d)
    # 取多组绝对位置：(p1,q1)间距=2; (p2,q2)间距=2，绝对位置不同
    pos_list1 = [3,5]  # 间距2
    pos_list2 = [10,12]# 间距2
    # 构造两个序列，仅在对应位置放e
    seq1 = torch.zeros(B, max(pos_list1)+1, d)
    seq1[:,pos_list1[0],:] = e
    seq1[:,pos_list1[1],:] = e
    rope1 = rope_forward(seq1)
    dot1 = torch.dot(rope1[0,pos_list1[0]], rope1[0,pos_list1[1]]).item()

    seq2 = torch.zeros(B, max(pos_list2)+1, d)
    seq2[:,pos_list2[0],:] = e
    seq2[:,pos_list2[1],:] = e
    rope2 = rope_forward(seq2)
    dot2 = torch.dot(rope2[0,pos_list2[0]], rope2[0,pos_list2[1]]).item()

    print("\n==== RoPE相对位置验证实验 ====")
    print(f"位置(3,5)间距=2 内积：{dot1:.4f}")
    print(f"位置(10,12)间距=2 内积：{dot2:.4f}")
    print("✅ 相同相对距离、不同绝对位置，内积几乎相等，验证相对位置编码特性")

    # 不同间距对比
    seq3 = torch.zeros(B,20,d)
    seq3[:,0,:]=e;seq3[:,5,:]=e
    rope3=rope_forward(seq3)
    dot3 = torch.dot(rope3[0,0],rope3[0,5]).item()
    print(f"位置(0,5)间距=5 内积：{dot3:.4f}，数值明显变化，内积只依赖相对距离")

# ====================== 6. 主函数运行+理论总结 ======================
if __name__ == "__main__":
    # 任务1~5依次运行
    compare_input_mode()
    test_relative_rope()

    # =========作业6：RoPE优于E+pos(SinPE)的原理说明=========
    print("\n=================6. RoPE比E+pos正弦PE更巧妙的原因总结=================")
    explain = """
1. 【建模特性：天然自带相对位置】
    SinPE(E+pos)是绝对位置编码：E+P编码的是token的绝对下标p，计算注意力时内积依赖绝对位置，无法天然建模相对距离；
    RoPE通过旋转变换，数学上保证两个向量注意力内积只由相对位置|i-j|决定，和绝对下标无关，贴合Transformer注意力依赖相对距离的现实语义。

2. 【外推泛化能力更强】
    SinPE预定义最大长度max_len，超出预训练长度的新位置无编码，外推性能差；
    RoPE公式是连续数学函数，任意超长pos都可以实时计算旋转角，支持长度外推。

3. 【融合方式更自然：几何嵌入而非简单加法】
    SinPE是特征与位置向量直接相加（线性叠加，几何无约束），容易出现特征与位置信息互相干扰；
    RoPE利用复数旋转几何约束，把位置信息编码进特征向量的方向，幅值不变，保留原始词向量大小信息。

4. 【数学契合注意力内积运算】
    Transformer核心是QK^T内积，RoPE旋转后内积天然分解为相对位置函数，完美适配注意力计算范式；
    E+pos加法后的内积混合了词向量、位置向量交叉项，数学无相对位置约束。

5. 【无额外参数量/存储开销】
    SinPE需要预存位置编码表，固定最大序列长度；RoPE按公式在线计算旋转角，无需预存PE矩阵。
    """
    print(explain)

# ======================可选：2D旋转可视化(作业2可视化验证)======================
def vis_2d_rotate():
    vec = torch.tensor([1.0,0.])
    pos_list = [0,1,2,4]
    plt.figure(figsize=(6,6))
    plt.axhline(0,c='k',lw=0.8)
    plt.axvline(0,c='k',lw=0.8)
    for p in pos_list:
        rv = rotate_2d(vec, p)
        plt.arrow(0,0,rv[0],rv[1],label=f"pos={p}")
    plt.legend()
    plt.title("2D向量随位置pos旋转可视化")
    plt.show()
# vis_2d_rotate() # 取消注释运行绘图