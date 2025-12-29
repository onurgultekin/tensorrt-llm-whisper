# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Upstream source:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/whisper/convert_checkpoint.py
#
# This file is intentionally kept close to upstream to simplify upgrades.
import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import save_file

import tensorrt_llm
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models.convert_utils import weight_only_quantize_dict
from tensorrt_llm.quantization import QuantAlgo


def _rename_trtllm018_weight_keys(weights: dict, *, component: str) -> dict:
    """Rename checkpoint keys to TRT-LLM 0.18+ Whisper naming scheme.

    TRT-LLM 0.18's Whisper `trtllm-build` expects keys like:
      - encoder: `encoder_layers.*`, `conv1.*`, `position_embedding.*`, `ln_post.*`
      - decoder: `decoder_layers.*`, `vocab_embedding.*`, `position_embedding.*`, ...

    This replaces the older `transformer.*` prefix naming used by earlier examples.
    We intentionally avoid writing both key sets, because safetensors rejects saving
    tensors that share memory under multiple keys.
    """

    out: dict = {}

    def _maybe_add_ln_aliases(k: str, v: torch.Tensor) -> None:
        # Some versions use `ln_f`, others use `ln_post`, and some decoder configs use
        # `final_layernorm`. These tensors are small, so cloning aliases is OK.
        if k.startswith("ln_f."):
            out.setdefault("ln_post." + k[len("ln_f.") :], v.clone())
            out.setdefault("final_layernorm." + k[len("ln_f.") :], v.clone())
            return
        if k.startswith("ln_post."):
            out.setdefault("ln_f." + k[len("ln_post.") :], v.clone())
            out.setdefault("final_layernorm." + k[len("ln_post.") :], v.clone())
            return
        if k.startswith("final_layernorm."):
            out.setdefault("ln_post." + k[len("final_layernorm.") :], v.clone())
            out.setdefault("ln_f." + k[len("final_layernorm.") :], v.clone())

    if component == "encoder":
        for k, v in weights.items():
            if not isinstance(k, str):
                continue

            if k.startswith("transformer.layers."):
                new_k = "encoder_layers." + k[len("transformer.layers.") :]
            elif k.startswith("transformer."):
                new_k = k[len("transformer.") :]
            else:
                new_k = k

            if new_k.startswith("ln_f."):
                new_k = "ln_post." + new_k[len("ln_f.") :]

            out[new_k] = v
            if isinstance(v, torch.Tensor):
                _maybe_add_ln_aliases(new_k, v)
        return out

    if component == "decoder":
        for k, v in weights.items():
            if not isinstance(k, str):
                continue

            if k.startswith("transformer.layers."):
                new_k = "decoder_layers." + k[len("transformer.layers.") :]
            elif k.startswith("transformer."):
                new_k = k[len("transformer.") :]
            else:
                new_k = k

            # TRT-LLM 0.18 decoder expects embedding.* and final_layernorm.*
            if new_k == "vocab_embedding.weight":
                new_k = "embedding.vocab_embedding.weight"
            elif new_k == "position_embedding.weight":
                new_k = "embedding.position_embedding.weight"
            elif new_k.startswith("ln_f."):
                new_k = "final_layernorm." + new_k[len("ln_f.") :]
            elif new_k.startswith("ln_post."):
                new_k = "final_layernorm." + new_k[len("ln_post.") :]

            out[new_k] = v
            if isinstance(v, torch.Tensor):
                _maybe_add_ln_aliases(new_k, v)
        return out

    raise ValueError(f"Unknown component: {component!r}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="assets")
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--model_name',
                        type=str,
                        default="large-v3",
                        choices=[
                            "large-v3-turbo",
                            "large-v3",
                            "large-v2",
                            "medium",
                            "small",
                            "base",
                            "tiny",
                            "medium.en",
                            "small.en",
                            "base.en",
                            "tiny.en",
                            "distil-large-v3",
                            "distil-large-v2",
                            "distil-medium.en",
                            "distil-small.en",
                        ])
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_gptq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    args = parser.parse_args()
    return args


def get_encoder_config(model_metadata: dict, dtype: str,
                       quant_algo: QuantAlgo) -> dict:
    model_is_multilingual = (model_metadata['n_vocab'] >= 51865)
    num_languages = model_metadata['n_vocab'] - 51765 - int(
        model_is_multilingual)
    return {
        'architecture': "WhisperEncoder",
        'dtype': dtype,
        'num_hidden_layers': model_metadata['n_audio_layer'],
        'num_attention_heads': model_metadata['n_audio_head'],
        'hidden_size': model_metadata['n_audio_state'],
        'max_position_embeddings': model_metadata['n_audio_ctx'],
        'has_position_embedding': True,
        'n_mels': model_metadata['n_mels'],
        'vocab_size': model_metadata['n_vocab'],
        'hidden_act': "gelu",
        'num_languages': num_languages,
        'quantization': {
            'quant_algo': quant_algo
        },
    }


def get_decoder_config(model_metadata: dict, dtype: str, logits_dtype: str,
                       quant_algo: QuantAlgo) -> dict:
    return {
        'architecture': "DecoderModel",
        'dtype': dtype,
        'logits_dtype': logits_dtype,
        'num_hidden_layers': model_metadata['n_text_layer'],
        'num_attention_heads': model_metadata['n_text_head'],
        'hidden_size': model_metadata['n_text_state'],
        'norm_epsilon': 1e-5,
        'vocab_size': model_metadata['n_vocab'],
        'hidden_act': "gelu",
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'max_position_embeddings': model_metadata['n_text_ctx'],
        'use_prompt_tuning': False,
        'head_size':
        model_metadata['n_text_state'] // model_metadata['n_text_head'],
        'has_position_embedding': True,
        'layernorm_type': LayerNormType.LayerNorm,
        'has_attention_qkvo_bias': True,
        'has_mlp_bias': True,
        'has_model_final_layernorm': True,
        'has_embedding_layernorm': False,
        'has_embedding_scale': False,
        'ffn_hidden_size': 4 * model_metadata['n_text_state'],
        'q_scaling': 1.0,
        'layernorm_position': LayerNormPositionType.pre_layernorm,
        'relative_attention': False,
        'max_distance': 0,
        'num_buckets': 0,
        'model_type': 'whisper',
        'rescale_before_lm_head': False,
        'encoder_hidden_size': model_metadata['n_text_state'],
        'encoder_num_heads': model_metadata['n_text_head'],
        'encoder_head_size': None,
        'skip_cross_kv': False,
        'quantization': {
            'quant_algo': quant_algo
        },
    }


def convert_openai_whisper_encoder(
    model_metadata: dict,
    model_params: dict,
    quant_algo: str = None,
):
    weights = {}

    def sinusoids(length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
            np.newaxis, :]
        return torch.cat([torch.sin(scaled_time),
                          torch.cos(scaled_time)],
                         dim=1)

    weights['transformer.position_embedding.weight'] = sinusoids(
        model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state']).contiguous()

    weights['transformer.conv1.weight'] = torch.unsqueeze(
        model_params['encoder.conv1.weight'], -1).contiguous()
    weights['transformer.conv1.bias'] = model_params[
        'encoder.conv1.bias'].contiguous()
    weights['transformer.conv2.weight'] = torch.unsqueeze(
        model_params['encoder.conv2.weight'], -1).contiguous()
    weights['transformer.conv2.bias'] = model_params[
        'encoder.conv2.bias'].contiguous()

    for i in range(model_metadata['n_audio_layer']):
        trtllm_layer_name_prefix = f'transformer.layers.{i}'

        weights[
            f'{trtllm_layer_name_prefix}.attention_layernorm.weight'] = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.weight'].contiguous()
        weights[
            f'{trtllm_layer_name_prefix}.attention_layernorm.bias'] = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.bias'].contiguous()

        t = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0).contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.qkv.weight'] = t

        bias_shape = model_params['encoder.blocks.' + str(i) +
                                  '.attn.query.bias'].shape
        dtype = model_params['encoder.blocks.' + str(i) +
                             '.attn.query.bias'].dtype
        fused_bias = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.bias'],
            torch.zeros([*bias_shape], dtype=dtype),
            model_params['encoder.blocks.' + str(i) + '.attn.value.bias']
        ],
                               dim=0).contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.qkv.bias'] = fused_bias

        t = model_params['encoder.blocks.' + str(i) +
                         '.attn.out.weight'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.dense.weight'] = t

        weights[f'{trtllm_layer_name_prefix}.attention.dense.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.attn.out.bias'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp_ln.weight'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp_ln.bias'].contiguous()

        t = model_params['encoder.blocks.' + str(i) +
                         '.mlp.0.weight'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = t
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp.0.bias'].contiguous()

        t = model_params['encoder.blocks.' + str(i) +
                         '.mlp.2.weight'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = t
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp.2.bias'].contiguous()

    weights['transformer.ln_f.weight'] = model_params[
        'encoder.ln_post.weight'].contiguous()
    weights['transformer.ln_f.bias'] = model_params[
        'encoder.ln_post.bias'].contiguous()

    weights = weight_only_quantize_dict(weights, quant_algo=quant_algo, plugin=True)
    return _rename_trtllm018_weight_keys(weights, component="encoder")


def convert_openai_whisper_decoder(model_metadata: dict,
                                   model_params: dict,
                                   quant_algo: str = None):
    weights = {}

    weights['transformer.vocab_embedding.weight'] = model_params[
        'decoder.token_embedding.weight']
    weights['transformer.position_embedding.weight'] = model_params[
        'decoder.positional_embedding']
    weights['lm_head.weight'] = model_params[
        'decoder.token_embedding.weight'].clone()

    for i in range(model_metadata['n_text_layer']):
        trtllm_layer_name_prefix = f'transformer.layers.{i}'

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0)
        weights[f'{trtllm_layer_name_prefix}.self_attention.qkv.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.attn.out.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.self_attention.dense.weight'] = t

        bias_shape = model_params['decoder.blocks.' + str(i) +
                                  '.attn.query.bias'].shape
        dtype = model_params['decoder.blocks.' + str(i) +
                             '.attn.query.bias'].dtype
        weights[
            f'{trtllm_layer_name_prefix}.self_attention.qkv.bias'] = torch.cat(
                [
                    model_params['decoder.blocks.' + str(i) +
                                 '.attn.query.bias'],
                    torch.zeros([*bias_shape], dtype=dtype),
                    model_params['decoder.blocks.' + str(i) +
                                 '.attn.value.bias']
                ],
                dim=0)
        weights[
            f'{trtllm_layer_name_prefix}.self_attention.dense.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.attn.out.bias']

        weights[
            f'{trtllm_layer_name_prefix}.self_attention_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.attn_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.self_attention_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.attn_ln.bias']

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.cross_attn.key.weight'],
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.value.weight']
        ],
                      dim=0)
        weights[f'{trtllm_layer_name_prefix}.cross_attention.qkv.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.out.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.cross_attention.dense.weight'] = t

        bias_shape = model_params['decoder.blocks.' + str(i) +
                                  '.cross_attn.query.bias'].shape
        dtype = model_params['decoder.blocks.' + str(i) +
                             '.cross_attn.query.bias'].dtype
        cross_attn_qkv_bias = torch.cat([
            model_params['decoder.blocks.' + str(i) + '.cross_attn.query.bias'],
            torch.zeros([*bias_shape], dtype=dtype),
            model_params['decoder.blocks.' + str(i) + '.cross_attn.value.bias']
        ],
                                        dim=0)

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention.qkv.bias'] = cross_attn_qkv_bias

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention.dense.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn.out.bias']

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.cross_attention_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn_ln.bias']

        t = model_params['decoder.blocks.' + str(i) +
                         '.mlp.0.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.mlp.2.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = t

        weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = model_params[
            'decoder.blocks.' + str(i) + '.mlp.0.bias']
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = model_params[
            'decoder.blocks.' + str(i) + '.mlp.2.bias']

        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.mlp_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.mlp_ln.bias']

    weights['transformer.ln_f.weight'] = model_params['decoder.ln.weight']
    weights['transformer.ln_f.bias'] = model_params['decoder.ln.bias']

    weights = weight_only_quantize_dict(weights, quant_algo=quant_algo, plugin=True)
    return _rename_trtllm018_weight_keys(weights, component="decoder")


if __name__ == "__main__":
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        quant_algo = QuantAlgo.W4A16
    elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        quant_algo = QuantAlgo.W4A16_GPTQ

    model_path = os.path.join(args.model_dir, args.model_name + '.pt')
    assert os.path.exists(model_path), f"Model {model_path} does not exist."

    model = torch.load(model_path, map_location='cpu')
    print(f"Loaded model from {model_path}")
    model_metadata = model['dims']
    model_state_dict = model['model_state_dict']
    for param_tensor in model_state_dict:
        model_state_dict[param_tensor] = model_state_dict[param_tensor].half()

    def convert_and_save(component: str = "encoder"):
        if component == "encoder":
            config = get_encoder_config(model_metadata, args.dtype, quant_algo)
        else:
            config = get_decoder_config(model_metadata, args.dtype,
                                        args.logits_dtype, quant_algo)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            config['quantization'].update({
                'has_zero_point': True,
            })

        component_save_dir = os.path.join(args.output_dir, component)
        if not os.path.exists(component_save_dir):
            os.makedirs(component_save_dir)

        with open(os.path.join(component_save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        if component == "encoder":
            weights = convert_openai_whisper_encoder(model_metadata,
                                                     model_state_dict,
                                                     quant_algo=quant_algo)
        else:
            assert component == "decoder"
            weights = convert_openai_whisper_decoder(model_metadata,
                                                     model_state_dict,
                                                     quant_algo=quant_algo)

        save_file(weights, os.path.join(component_save_dir,
                                        f'rank0.safetensors'))

    print("Converting encoder checkpoints...")
    convert_and_save("encoder")
    print("Converting decoder checkpoints...")
    convert_and_save("decoder")

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
