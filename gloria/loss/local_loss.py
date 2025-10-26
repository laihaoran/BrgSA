import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the class to compute local loss
class LocalLossCalculator(nn.Module):
    def __init__(self, num_blocks=4, num_phrases=3, top_k=2, tau_local=0.07):
        super(LocalLossCalculator, self).__init__()
        self.num_blocks = num_blocks
        self.num_phrases = num_phrases
        self.top_k = top_k
        self.tau_local = tau_local

    def forward(self, patch_feats, patch_attn_map, word_feats, word_attn_map):
        batch_size = patch_feats.size(0)
        num_patches = patch_feats.size(1)
        num_words = word_feats.size(1)
        feat_dim = patch_feats.size(-1)

        # Initialize tensors to store block and phrase features
        block_feats = torch.zeros(batch_size, self.num_blocks, feat_dim, device=patch_feats.device)
        phrase_feats = torch.zeros(batch_size, self.num_phrases, feat_dim, device=word_feats.device)

        # Process each sample individually
        for i in range(batch_size):
            # Aggregate image patches into blocks
            # Compute attention weights for patches
            attn_weights_patches = patch_attn_map[i].mean(dim=1)  # Shape: (num_patches)
            attn_weights_patches = attn_weights_patches.clone()  # To avoid modifying the original attention map

            for block_idx in range(self.num_blocks):
                k_patch = num_patches // self.num_blocks

                # Select top-k patches based on attention weights
                topk_values, topk_indices = torch.topk(attn_weights_patches, k_patch)
                selected_feats = patch_feats[i, topk_indices]  # Shape: (k_patch, feat_dim)
                block_feat = selected_feats.mean(dim=0)  # Shape: (feat_dim)
                block_feats[i, block_idx] = block_feat
                attn_weights_patches[topk_indices] = -float('inf')  # Avoid re-selection

            # Aggregate words into phrases
            # Compute attention weights for words
            attn_weights_words = word_attn_map[i].mean(dim=1)  # Shape: (num_words)
            attn_weights_words = attn_weights_words.clone()  # To avoid modifying the original attention map

            for phrase_idx in range(self.num_phrases):
                k_word = num_words // self.num_phrases

                # Select top-k words based on attention weights
                topk_values, topk_indices = torch.topk(attn_weights_words, k_word)
                selected_feats = word_feats[i, topk_indices]  # Shape: (k_word, feat_dim)
                phrase_feat = selected_feats.mean(dim=0)  # Shape: (feat_dim)
                phrase_feats[i, phrase_idx] = phrase_feat
                attn_weights_words[topk_indices] = -float('inf')  # Avoid re-selection

        # Normalize features
        block_feats = F.normalize(block_feats, dim=-1)
        phrase_feats = F.normalize(phrase_feats, dim=-1)

        # Calculate local loss
        return self.calculate_loss(block_feats, phrase_feats)

    def calculate_loss(self, block_feats, phrase_feats):
        batch_size = block_feats.size(0)
        selected_blocks = []
        selected_phrases = []

        # For each sample, find the top-k similar pairs and select one randomly
        for i in range(batch_size):
            similarity_matrix = torch.matmul(block_feats[i], phrase_feats[i].T)  # Shape: (num_blocks, num_phrases)
            similarity_flat = similarity_matrix.view(-1)  # Flatten to (num_blocks * num_phrases)
            topk_values, topk_indices = similarity_flat.topk(self.top_k)
            random_idx = torch.randint(0, self.top_k, (1,))
            selected_pair_idx = topk_indices[random_idx]
            block_index = selected_pair_idx // self.num_phrases
            phrase_index = selected_pair_idx % self.num_phrases
            selected_blocks.append(block_feats[i, block_index])
            selected_phrases.append(phrase_feats[i, phrase_index])

        # Stack the selected blocks and phrases
        selected_blocks = torch.cat(selected_blocks, dim=0)  # Shape: (batch_size, feat_dim)
        selected_phrases = torch.cat(selected_phrases, dim=0)  # Shape: (batch_size, feat_dim)

        # Normalize selected features again (optional)
        selected_blocks = F.normalize(selected_blocks, dim=-1)
        selected_phrases = F.normalize(selected_phrases, dim=-1)

        # Compute global similarity matrix
        global_similarity_matrix = torch.matmul(selected_blocks, selected_phrases.T) / self.tau_local  # Shape: (batch_size, batch_size)

        # Create labels
        labels = torch.arange(batch_size, device=block_feats.device)

        # Compute losses in both directions
        loss_block_to_phrase = F.cross_entropy(global_similarity_matrix, labels)
        loss_phrase_to_block = F.cross_entropy(global_similarity_matrix.T, labels)

        # Average the losses
        loss = 0.5 * (loss_block_to_phrase + loss_phrase_to_block)

        return loss
