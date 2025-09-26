class SwinUNETR(nn.Module):
  def __init__(self,img_size=(128,256,256),in_channels=3,num_classes=14,feature_size=48,depths=(2, 2, 2, 2),num_heads=(3, 6, 12, 24),window_size=7,
               drop_rate=0.0,
               attn_drop_rate=0.0,
               use_3d=False,
               use_checkpoint=False):
    super().__init__()
    self.use_3d = use_3d
    self.num_classes = num_classes
    self.depths = depths  # FIXED: Store depths as instance variable
    spatial_dims = 3 if use_3d else 2
    
    # Ensure window_size and patch_size are tuples
    patch_size = (4, 4, 4) if use_3d else (4, 4)
    if isinstance(window_size, int):
        window_size = tuple([window_size] * spatial_dims)

    self.patch_embed = PatchEmbedding(
            patch_size=4, # MONAI's PatchEmbedding uses an int
            in_chans=in_channels,
            embed_dim=feature_size,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims
        )
    
    if self.use_3d:
      self.patch_grid_size = (img_size[0]//4, img_size[1]//4, img_size[2]//4)
      current_resolution = self.patch_grid_size
    else:
      self.patch_grid_size = (img_size[0]//4, img_size[1]//4)
      current_resolution = self.patch_grid_size
      
    self.encoder_layers = nn.ModuleList()
    self.encoder_features = []

    embed_dim = feature_size
    for i in range(len(depths)):
            stage_blocks = nn.ModuleList()
            for j in range(depths[i]):
                if self.use_3d:
                    block = SwinTransformerBlock3D(
                        dim=embed_dim * (2 ** i),
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=tuple(s // 2 for s in window_size) if (j % 2 != 0) else (0,0,0),
                        mlp_ratio=4, qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                    )
                else: # Original 2D Block
                    block = SwinTransformerBlock(
                        dim=embed_dim * (2 ** i),
                        num_heads=num_heads[i],
                        window_size=window_size[0],
                        shift_size=0 if (j % 2 == 0) else window_size[0] // 2,
                        input_resolution=current_resolution
                    )
                stage_blocks.append(block)

            self.encoder_layers.append(stage_blocks)
            self.encoder_features.append(embed_dim * (2 ** i))

            if i < len(depths) - 1:
              if self.use_3d:
                  downsample = PatchMerging3D(input_resolution=current_resolution, dim=embed_dim * (2 ** i))
                  current_resolution = (current_resolution[0]//2, current_resolution[1]//2, current_resolution[2]//2)
              else:
                  downsample = PatchMerging(input_resolution=current_resolution, dim=embed_dim * (2 ** i))
                  current_resolution = (current_resolution[0]//2, current_resolution[1]//2)
              
              self.encoder_layers.append(downsample)

    self.final_resolution = current_resolution
    self.final_channels = embed_dim * (2 ** (len(depths) - 1))
    
    Conv = nn.Conv3d if use_3d else nn.Conv2d
    self.to_conv = nn.Sequential(
            nn.LayerNorm(self.final_channels),
            nn.Linear(self.final_channels, self.final_channels)
        )
    
    # --- Decoder setup (remains the same as it was already 3D-aware) ---
    self.decoders = nn.ModuleList()
    decoder_channels = self.encoder_features[::-1]

    for i in range(len(depths)-1):
      in_ch = decoder_channels[i]
      skip_ch = decoder_channels[i + 1] if i < len(depths) - 1 else 0
      out_ch = decoder_channels[i + 1] if i < len(depths) - 1 else feature_size
      decoder = UNetDecoder(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch, use_3d=use_3d)
      self.decoders.append(decoder)

    self.final_decoder = UNetDecoder(in_channels=feature_size, skip_channels=0, out_channels=feature_size, use_3d=use_3d)
    
    # --- Head setup (remains the same) ---
    self.head = nn.Sequential(
            Conv(feature_size, feature_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_size // 2) if use_3d else nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(inplace=True),
            Conv(feature_size // 2, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool3d(1) if use_3d else nn.AdaptiveAvgPool2d(1)
        )
    self._init_weights()


  def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

  def forward_encoder(self, x):
      B = x.shape[0]
      features = []
      
      x = self.patch_embed(x)
      current_res = self.patch_grid_size

      layer_idx = 0
      for stage_idx in range(len(self.depths)):  # Now self.depths exists
          stage_blocks = self.encoder_layers[layer_idx]
          if self.use_3d:
              D, H, W = current_res
              for block in stage_blocks:
                  x = block(x, D, H, W)
          else:
              H, W = current_res
              for block in stage_blocks:
                  x = block(x, H=H, W=W) # Pass H,W to original block
          
          features.append((x.clone(), current_res))
          layer_idx += 1
          
          if stage_idx < len(self.depths) - 1:
              downsample = self.encoder_layers[layer_idx]
              x = downsample(x)
              if self.use_3d:
                  current_res = (current_res[0] // 2, current_res[1] // 2, current_res[2] // 2)
              else:
                  current_res = (current_res[0] // 2, current_res[1] // 2)
              layer_idx += 1
              
      return x, features

  def forward(self,x):
      batch_size = x.shape[0]
      encoded, skip_features = self.forward_encoder(x)
      
      x = self.to_conv(encoded)
      
      # Reshape for decoder
      if self.use_3d:
          D, H, W = self.final_resolution
          x = x.view(batch_size, D, H, W, -1).permute(0, 4, 1, 2, 3)
      else:
          H, W = self.final_resolution
          x = x.view(batch_size, H, W, -1).permute(0, 3, 1, 2)

      skip_idx = len(skip_features) - 2 # Start from the second to last skip
      for decoder in self.decoders:
        if skip_idx >= 0:
          skip_x, skip_res = skip_features[skip_idx]
          if self.use_3d:
              skip_D, skip_H, skip_W = skip_res
              skip = skip_x.view(batch_size, skip_D, skip_H, skip_W, -1).permute(0, 4, 1, 2, 3)
          else:
              skip_H, skip_W = skip_res
              skip = skip_x.view(batch_size, skip_H, skip_W, -1).permute(0, 3, 1, 2)
          x = decoder(x, skip)
          skip_idx -= 1
        else:
          x = decoder(x, None)

      x = self.final_decoder(x)
      x = self.head(x)
      x = x.view(batch_size, self.num_classes)
      return x
