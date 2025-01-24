import torch
import torch.nn.functional as F

def sample_vectors(f_v, k):
    """
    Uniformly sample k vectors from f_v.
    """
    t = f_v.size(0)
    indices = torch.linspace(0, t - 1, steps=k).long()
    sampled_vectors = f_v[indices]
    return sampled_vectors, indices

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two tensors a and b.
    """
    return F.cosine_similarity(a, b, dim=-1)

def calculate_attention(f_qst, f_v_sampled):
    """
    Calculate attention values
    """
    q = f_qst.unsqueeze(0)  # Shape: [1, d]
    k = f_v_sampled         # Shape: [k, d]
    v = f_v_sampled         # Shape: [k, d]

    # Attention calculation: att_weights = softmax(q * k^T / sqrt(d))
    d = f_qst.size(-1)
    att_weights = F.softmax(torch.matmul(q, k.T) / (d ** 0.5), dim=-1)
    att_f_v = torch.matmul(att_weights, v)  # Shape: [1, d]
    return att_f_v.squeeze(0)  # Shape: [d]

def process_numbers(number_list):
    unique_numbers = sorted(set(int(num) for num in number_list))
    return unique_numbers

def iterative_sampling(f_v, f_text, k, m, a1, a2):
    """
    Perform the iterative sampling process.
    """
    t, d = f_v.shape
    indices_record = []
    iter_samples = [0]


    for _ in range(m):
        # Uniformly sample k vectors from f_v
        f_v_sampled, sampled_indices = sample_vectors(f_v, k)
        sampled_indices += sum(iter_samples)

        # Compute cosine similarity between consecutive vectors
        sim1 = cosine_similarity(f_v_sampled[:-1], f_v_sampled[1:])
        sim1 = torch.cat([sim1, sim1[-1].unsqueeze(0)])  # Ensure last and second-last are the same

        # Calculate att_f_v
        att_f_v = torch.stack([calculate_attention(f_text, f_v_sampled[i].unsqueeze(0)) for i in range(k)])

        # Compute cosine similarity between consecutive att_f_v vectors
        sim2 = cosine_similarity(att_f_v[:-1], att_f_v[1:])
        sim2 = torch.cat([sim2, sim2[-1].unsqueeze(0)])  # Ensure last and second-last are the same

        # Sum and find the max index
        sim = a1 * sim1 + a2 * sim2
        max_sim_index = torch.argmax(sim)

        # Use max_sim_index as center, select new vectors
        center_idx = sampled_indices[max_sim_index]
        start_idx = max(0, center_idx - t // k)
        end_idx = min(t, center_idx + t // k)
        iter_samples.append(start_idx)
        f_v = f_v[start_idx:end_idx]
        indices_record += list(sampled_indices)
    indices_record = process_numbers(indices_record)

    return indices_record