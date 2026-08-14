import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChronosRelationEncoder(nn.Module):
    """Chronos encoder used to produce channel embeddings for Top-K retrieval.

    Defaults to the frozen black-box setup RAF uses. With finetune=True the T5
    encoder is registered as a submodule so its weights appear in
    named_parameters()/state_dict() and receive gradients from the Stage-2
    retrieval loss. Gradients reach the weights but never the input series:
    Chronos tokenizes real values into discrete bins, so the input->token_ids
    map is not differentiable.
    """

    _DTYPES = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    def __init__(self, model_id, embedding_dim=768, dtype='bfloat16', random_init=False,
                 finetune=False, grad_checkpointing=True, pooling='mean'):
        super().__init__()
        if dtype not in self._DTYPES:
            raise ValueError(f'Unsupported Chronos dtype: {dtype}')
        if pooling not in ('mean', 'eos'):
            raise ValueError(f'Unsupported chronos_pooling: {pooling}')
        # 'mean' drops the EOS token and averages the value tokens, which is what
        # this repo has always done. 'eos' keeps only the EOS token, the summary
        # position TS-RAG retrieves with (embeddings[:, -1, :]).
        self.pooling = pooling
        self.model_id = model_id
        self.embedding_dim = int(embedding_dim)
        self.finetune = bool(finetune)
        if self.finetune and dtype != 'float32':
            # Adam on bf16/fp16 master weights is unstable; the frozen path can
            # keep the low-precision weights because it never takes a step.
            print(f'[chronos] finetune=1 overrides chronos_dtype={dtype} with float32')
            dtype = 'float32'
        self.dtype = self._DTYPES[dtype]
        self.random_init = bool(random_init)
        self.grad_checkpointing = bool(grad_checkpointing) and self.finetune
        self.pipeline = None
        self.device = None
        self._context_logged = False

    def _load(self, device):
        if self.pipeline is not None:
            # When fine-tuning, the T5 is a registered submodule, so nn.Module.to()
            # may legitimately have moved it after the initial load.
            if self.finetune:
                self.device = device
                return
            if self.device != device:
                raise RuntimeError(
                    f'Chronos was loaded on {self.device}, but embeddings were requested on {device}'
                )
            return
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ImportError(
                'Chronos retrieval requires the chronos-forecasting package. '
                'Install it in the active environment before running this experiment.'
            ) from exc

        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=str(device),
            torch_dtype=self.dtype,
        )
        self.device = device
        if self.random_init:
            t5_model = self.pipeline.model.model
            with torch.no_grad():
                # Call the model-specific initializer directly because
                # from_pretrained marks loaded modules as already initialized.
                t5_model.apply(t5_model._init_weights)
                t5_model.tie_weights()
            print(f'[chronos] randomly initialized checkpoint architecture: {self.model_id}')
        if self.finetune:
            # Only the encoder is ever run (ChronosPipeline.encode calls
            # model.encoder), so registering the whole seq2seq would put ~160
            # decoder tensors in the optimizer that can never receive a
            # gradient, and pay Adam state for them.
            for parameter in self.pipeline.model.model.parameters():
                parameter.requires_grad = False
            # Registering the encoder puts its parameters in named_parameters()
            # and state_dict(), which is what makes them reachable by the optimizer.
            self.t5_encoder = self.pipeline.model.model.encoder
            self.t5_encoder.train()
            for parameter in self.t5_encoder.parameters():
                parameter.requires_grad = True
            if self.grad_checkpointing:
                # A [batch*channel, seq] encode keeps every layer's attention
                # activations alive until backward; at seq_len 336 with
                # batch_size 32 that alone exceeds 79 GiB.
                self.pipeline.model.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False}
                )
            print(
                f'[chronos] fine-tuning enabled: {self.model_id} '
                f'(encoder only, grad_checkpointing={int(self.grad_checkpointing)})'
            )
        else:
            self.pipeline.model.model.eval()
            for parameter in self.pipeline.model.model.parameters():
                parameter.requires_grad = False

        actual_dim = int(self.pipeline.model.model.config.d_model)
        if actual_dim != self.embedding_dim:
            raise RuntimeError(
                f'Chronos embedding dimension mismatch: configured={self.embedding_dim}, '
                f'checkpoint={actual_dim}'
            )

    def _grad_context(self):
        return contextlib.nullcontext() if self.finetune else torch.no_grad()

    def encode_channel_tokens(self, x):
        """Encode [batch, time, channel] into aligned token-level channel features."""
        device = x.device
        self._load(device)
        batch_size, seq_len, channels = x.shape
        context = x.transpose(1, 2).reshape(batch_size * channels, seq_len)

        # Tokenization is a non-differentiable quantization, so it always runs
        # detached; gradients enter through the encoder weights below.
        with torch.no_grad():
            token_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(
                context.detach().float().cpu()
            )
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        with self._grad_context():
            hidden = self.pipeline.model.encode(token_ids, attention_mask).float()
        # Chronos appends EOS after the value tokens. 'mean' pooling aligns on the
        # time tokens only and drops it; 'eos' keeps just that summary token, so
        # the masked mean downstream runs over a length-1 sequence and returns it.
        if self.pooling == 'eos':
            hidden = hidden[:, -1:]
            attention_mask = attention_mask[:, -1:]
        else:
            hidden = hidden[:, :-1]
            attention_mask = attention_mask[:, :-1]
        token_length = hidden.size(1)
        hidden = hidden.reshape(batch_size, channels, token_length, self.embedding_dim)
        attention_mask = attention_mask.reshape(batch_size, channels, token_length).bool()

        if not self._context_logged:
            configured_context = int(self.pipeline.model.config.context_length)
            effective_context = min(seq_len, configured_context)
            print(
                f'[chronos] model={self.model_id} random_init={int(self.random_init)} '
                f'input_length={seq_len} '
                f'effective_context={effective_context} token_length={token_length} '
                f'embedding_dim={self.embedding_dim}'
            )
            self._context_logged = True
        return hidden, attention_mask

    def encode_channels(self, x):
        """Encode [batch, time, channel] into normalized pooled channel features."""
        hidden, attention_mask = self.encode_channel_tokens(x)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        pooled = F.normalize(pooled.float(), dim=-1)
        return pooled
