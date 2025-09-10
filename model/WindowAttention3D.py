class WindowAttention3D(nn.Module):
  def __init__(self,dim,window_size,num_heads,qkv_bias=True, attn_drop=0., proj_drop=0.):
    super().__init__()
    self.dim = dim
    self.window_size = window_size
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self,x):
    B , N , C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
