from diffusers.models.attention_processor import AttnProcessor2_0,Attention
from diffusers import DiffusionPipeline
import torch

CACHED_ATTN_WEIGHTS="cached_attn_weights"

class CacheAttentionProcessor(AttnProcessor2_0):
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        attn_weight = (query @ key.transpose(-2, -1)) * attn.scale
        if attention_mask is not None:
            attn_weight = attn_weight + attention_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        setattr(attn, CACHED_ATTN_WEIGHTS, attn_weight.detach())

        hidden_states = attn_weight @ value

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
def get_modules_of_types(model, target_classes):
        return [(name, module) for name, module in model.named_modules()
                if isinstance(module, target_classes)]

def insert_cache_attn(pipe:DiffusionPipeline):
    attn_list=get_modules_of_types(pipe.unet,Attention)
    for name,module in attn_list:
        if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==AttnProcessor2_0:
            cache_processor=CacheAttentionProcessor()
            setattr(module,"processor",cache_processor)
            processor_name=name+".processor"
            pipe.unet.attn_processors[processor_name]=cache_processor