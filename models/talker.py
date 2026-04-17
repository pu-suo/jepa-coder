"""
Talker model — translates Reasoner plan indices into Python code.
Specification: docs/contract_3_talker_interface.md Sections 2-3

The Talker is an encoder-decoder Transformer:
  - Encoder processes concatenated [problem_embeds; plan_embeds] with segment IDs
  - Decoder cross-attends to encoder output, generates code autoregressively

IMPORTANT: plan_embedding is a SEPARATE nn.Embedding(512, 768), NOT the VQ
codebook. The Talker learns its own representation of each codebook index.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Talker(nn.Module):
    """
    Encoder-decoder Transformer that converts (problem_tokens, plan_indices)
    into Python code.

    Architecture (contract_3 Section 3.1):
        text_embedding:     nn.Embedding(vocab_size, 768)
        plan_embedding:     nn.Embedding(512, 768)       — NOT the VQ codebook
        segment_embedding:  nn.Embedding(2, 768)          — 0=problem, 1=plan
        position_embedding: nn.Embedding(max_seq_len, 768)
        encoder:            TransformerEncoder(4 layers, 768d, 12 heads)
        decoder:            TransformerDecoder(4 layers, 768d, 12 heads)
        lm_head:            nn.Linear(768, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # --- Embeddings ---
        # Separate embedding for problem text tokens
        self.text_embedding = nn.Embedding(vocab_size, dim)

        # Separate embedding for plan indices — NOT the VQ codebook.
        # The Talker learns its own representation of each codebook index.
        # 513 entries: indices 0-511 for codebook, index 512 reserved for
        # padding (so pad doesn't contaminate codebook index 0 = VQ STOP).
        self.plan_pad_id = 512
        self.plan_embedding = nn.Embedding(513, dim, padding_idx=512)

        # Segment embeddings: 0 = problem, 1 = plan
        self.segment_embedding = nn.Embedding(2, dim)

        # Positional embeddings (shared by encoder input and decoder input)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # --- Encoder (self-attention over [problem; plan]) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
        )

        # --- Decoder (self-attention + cross-attention to encoder output) ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
        )

        # --- Output projection ---
        self.lm_head = nn.Linear(dim, vocab_size)

    # ------------------------------------------------------------------
    # Encoder input construction (contract_3 Section 3.2)
    # ------------------------------------------------------------------

    def build_encoder_input(
        self,
        problem_tokens: torch.Tensor,
        plan_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct the encoder input by concatenating problem and plan embeddings
        with segment and position embeddings.

        Args:
            problem_tokens: LongTensor (B, L_prob) — problem text token IDs
            plan_indices:   LongTensor (B, M) — VQ codebook indices

        Returns:
            encoder_input: FloatTensor (B, L_prob + M, dim)
        """
        B, L_prob = problem_tokens.shape
        M = plan_indices.shape[1]
        device = problem_tokens.device

        # Embed problem tokens and plan indices separately
        prob_embeds = self.text_embedding(problem_tokens)    # (B, L_prob, dim)
        plan_embeds = self.plan_embedding(plan_indices)      # (B, M, dim)

        # Concatenate along sequence dimension
        combined = torch.cat([prob_embeds, plan_embeds], dim=1)  # (B, L_prob + M, dim)

        # Segment embeddings: 0 for problem positions, 1 for plan positions
        prob_segments = torch.zeros(B, L_prob, dtype=torch.long, device=device)
        plan_segments = torch.ones(B, M, dtype=torch.long, device=device)
        segments = torch.cat([prob_segments, plan_segments], dim=1)  # (B, L_prob + M)
        combined = combined + self.segment_embedding(segments)

        # Position embeddings over the full concatenated sequence
        total_len = L_prob + M
        positions = torch.arange(total_len, device=device).unsqueeze(0)  # (1, L_prob + M)
        combined = combined + self.position_embedding(positions)

        return combined

    # ------------------------------------------------------------------
    # Training forward pass (contract_3 Section 4.2)
    # ------------------------------------------------------------------

    def forward(
        self,
        problem_tokens: torch.Tensor,
        plan_indices: torch.Tensor,
        target_code: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing for training.

        Args:
            problem_tokens:      LongTensor (B, L_prob) — problem token IDs
            plan_indices:        LongTensor (B, M) — plan codebook indices
            target_code:         LongTensor (B, L_code) — target code token IDs
                                 (includes BOS prefix; labels are target_code[:, 1:])
            src_key_padding_mask: BoolTensor (B, L_prob + M) — True at pad positions
            tgt_key_padding_mask: BoolTensor (B, L_code - 1) — True at pad positions

        Returns:
            logits: FloatTensor (B, L_code - 1, vocab_size)
        """
        # Encode [problem; plan]
        encoder_input = self.build_encoder_input(problem_tokens, plan_indices)
        memory = self.encoder(
            encoder_input,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, L_prob + M, dim)

        # Decoder input: all target tokens except the last (teacher forcing)
        target_input = target_code[:, :-1]  # (B, L_code - 1)
        L_dec = target_input.shape[1]

        # Embed decoder input with position embeddings
        decoder_embeds = self.text_embedding(target_input)  # (B, L_dec, dim)
        dec_positions = torch.arange(L_dec, device=target_input.device).unsqueeze(0)
        decoder_embeds = decoder_embeds + self.position_embedding(dec_positions)

        # Causal mask for decoder self-attention
        causal_mask = torch.triu(
            torch.ones(L_dec, L_dec, device=target_input.device) * float('-inf'),
            diagonal=1,
        )  # (L_dec, L_dec)

        # Decoder forward with cross-attention to encoder memory
        decoder_out = self.decoder(
            decoder_embeds,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, L_dec, dim)

        # Project to vocabulary
        logits = self.lm_head(decoder_out)  # (B, L_dec, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Autoregressive generation (contract_3 Section 3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        problem_tokens: torch.Tensor,
        plan_indices: torch.Tensor,
        max_length: int = 1024,
    ) -> List[int]:
        """
        Generate code token IDs autoregressively.

        Args:
            problem_tokens: LongTensor (L_prob,) — problem text token IDs
            plan_indices:   LongTensor (M,) — VQ codebook indices

        Returns:
            List[int] — generated token IDs (BOS stripped, EOS stripped)
        """
        self.eval()
        device = problem_tokens.device

        # Add batch dimension for encoder
        prob_batch = problem_tokens.unsqueeze(0)    # (1, L_prob)
        plan_batch = plan_indices.unsqueeze(0)       # (1, M)

        # Encode [problem; plan]
        encoder_input = self.build_encoder_input(prob_batch, plan_batch)
        memory = self.encoder(encoder_input)  # (1, L_prob + M, dim)

        # Start with BOS token
        generated = [self.bos_token_id]

        for _ in range(max_length):
            gen_tokens = torch.tensor(
                [generated], dtype=torch.long, device=device,
            )  # (1, step+1)

            # Embed with positions
            gen_embeds = self.text_embedding(gen_tokens)  # (1, step+1, dim)
            positions = torch.arange(len(generated), device=device).unsqueeze(0)
            gen_embeds = gen_embeds + self.position_embedding(positions)

            # Causal mask
            seq_len = len(generated)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1,
            )

            # Decoder forward
            decoder_out = self.decoder(
                gen_embeds,
                memory,
                tgt_mask=causal_mask,
            )  # (1, step+1, dim)

            # Predict next token (greedy)
            logits = self.lm_head(decoder_out[:, -1, :])  # (1, vocab_size)
            next_token = logits.argmax(dim=-1).item()

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)

        # Return generated tokens without BOS
        return generated[1:]
