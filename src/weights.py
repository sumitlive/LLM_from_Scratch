import torch

def load_weights_from_hf(model, hf_state_dict):
    """
    Maps Hugging Face GPT-2 weights into our custom GPTModel architecture.
    Handles the transposition required because HF uses Conv1D instead of Linear for Attention/MLP.
    """
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone())

    # Embedding Layers
    model.pos_emb.weight = assign(model.pos_emb.weight, hf_state_dict['wpe.weight'])
    model.tok_emb.weight = assign(model.tok_emb.weight, hf_state_dict['wte.weight'])

    # Transformer Blocks
    for b in range(len(model.trf_blocks)):
        # Attention
        c_attn_w = hf_state_dict[f'h.{b}.attn.c_attn.weight']
        c_attn_b = hf_state_dict[f'h.{b}.attn.c_attn.bias']
        q_w, k_w, v_w = torch.tensor_split(c_attn_w, 3, dim=-1)
        q_b, k_b, v_b = torch.tensor_split(c_attn_b, 3, dim=-1)
        
        # NOTE: HF's Conv1D weights are transposed compared to standard PyTorch nn.Linear
        model.trf_blocks[b].att.W_query.weight = assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
        model.trf_blocks[b].att.W_key.weight = assign(model.trf_blocks[b].att.W_key.weight, k_w.T)
        model.trf_blocks[b].att.W_value.weight = assign(model.trf_blocks[b].att.W_value.weight, v_w.T)
        
        model.trf_blocks[b].att.W_query.bias = assign(model.trf_blocks[b].att.W_query.bias, q_b)
        model.trf_blocks[b].att.W_key.bias = assign(model.trf_blocks[b].att.W_key.bias, k_b)
        model.trf_blocks[b].att.W_value.bias = assign(model.trf_blocks[b].att.W_value.bias, v_b)
        
        c_proj_w = hf_state_dict[f'h.{b}.attn.c_proj.weight']
        c_proj_b = hf_state_dict[f'h.{b}.attn.c_proj.bias']
        model.trf_blocks[b].att.out_proj.weight = assign(model.trf_blocks[b].att.out_proj.weight, c_proj_w.T)
        model.trf_blocks[b].att.out_proj.bias = assign(model.trf_blocks[b].att.out_proj.bias, c_proj_b)
        
        # FeedForward (MLP)
        c_fc_w = hf_state_dict[f'h.{b}.mlp.c_fc.weight']
        c_fc_b = hf_state_dict[f'h.{b}.mlp.c_fc.bias']
        model.trf_blocks[b].ff.layers[0].weight = assign(model.trf_blocks[b].ff.layers[0].weight, c_fc_w.T)
        model.trf_blocks[b].ff.layers[0].bias = assign(model.trf_blocks[b].ff.layers[0].bias, c_fc_b)
        
        c_proj_w = hf_state_dict[f'h.{b}.mlp.c_proj.weight']
        c_proj_b = hf_state_dict[f'h.{b}.mlp.c_proj.bias']
        model.trf_blocks[b].ff.layers[2].weight = assign(model.trf_blocks[b].ff.layers[2].weight, c_proj_w.T)
        model.trf_blocks[b].ff.layers[2].bias = assign(model.trf_blocks[b].ff.layers[2].bias, c_proj_b)
        
        # Layer Norms
        model.trf_blocks[b].norm1.scale = assign(model.trf_blocks[b].norm1.scale, hf_state_dict[f'h.{b}.ln_1.weight'])
        model.trf_blocks[b].norm1.shift = assign(model.trf_blocks[b].norm1.shift, hf_state_dict[f'h.{b}.ln_1.bias'])
        model.trf_blocks[b].norm2.scale = assign(model.trf_blocks[b].norm2.scale, hf_state_dict[f'h.{b}.ln_2.weight'])
        model.trf_blocks[b].norm2.shift = assign(model.trf_blocks[b].norm2.shift, hf_state_dict[f'h.{b}.ln_2.bias'])

    # Final Layer Norm and Output Head
    model.final_norm.scale = assign(model.final_norm.scale, hf_state_dict['ln_f.weight'])
    model.final_norm.shift = assign(model.final_norm.shift, hf_state_dict['ln_f.bias'])
    model.out_head.weight = assign(model.out_head.weight, hf_state_dict['wte.weight'])

def download_and_load_gpt2(model, model_size="355M"):
    """
    Downloads GPT-2 weights from Hugging Face and loads them into the custom model.
    """
    try:
        from transformers import GPT2Model
    except ImportError:
        raise ImportError("Please install the transformers library: pip install transformers")
        
    print(f"Downloading {model_size} model weights from Hugging Face...")
    model_name = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl"
    }[model_size]
    
    hf_model = GPT2Model.from_pretrained(model_name)
    hf_state_dict = hf_model.state_dict()
    
    print("Mapping weights to custom architecture...")
    load_weights_from_hf(model, hf_state_dict)
    print("Weights loaded successfully!")
