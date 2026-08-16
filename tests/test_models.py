"""Model shape tests. All skipped automatically if torch isn't installed
(see conftest.requires_torch) — see README for how to run these locally."""
from conftest import requires_torch


@requires_torch
def test_sar_encoder_output_shape():
    import torch

    from src.models.sar_encoder import SAREncoder

    enc = SAREncoder(in_channels=2, image_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    x = torch.randn(3, 2, 32, 32)
    out = enc(x)
    assert out.shape == (3, (32 // 8) ** 2 + 1, 32)


@requires_torch
def test_eo_encoder_output_shape():
    import torch

    from src.models.eo_encoder import EOEncoder

    enc = EOEncoder(in_channels=4, image_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    x = torch.randn(3, 4, 32, 32)
    out = enc(x)
    assert out.shape == (3, (32 // 8) ** 2 + 1, 32)


@requires_torch
def test_all_fusion_types_output_correct_shape():
    import torch

    from src.models.fusion import build_fusion

    B, D, H = 4, 16, 24
    sar_pooled = torch.randn(B, D)
    eo_pooled = torch.randn(B, D)
    sar_tokens = torch.randn(B, 5, D)
    eo_tokens = torch.randn(B, 5, D)

    for ftype in ["concat", "gated", "none_eo", "none_sar"]:
        fusion = build_fusion(ftype, embed_dim=D, hidden_dim=H)
        if ftype in ("none_eo", "none_sar"):
            out = fusion(sar_pooled)
        else:
            out = fusion(sar_pooled, eo_pooled)
        assert out.shape == (B, H), f"{ftype} produced wrong shape {out.shape}"

    cross = build_fusion("cross_attention", embed_dim=D, hidden_dim=H)
    out = cross(sar_tokens, eo_tokens)
    assert out.shape == (B, H)


@requires_torch
def test_geosar_vlm_forward_pass_all_fusion_types(synthetic_processed_dir):
    import torch

    from src.data.dataset import GeoSARDataset
    from src.models.geosar_vlm import GeoSARVLM
    from src.utils.config import load_config
    from src.utils.tokenizer import SimpleTokenizer

    cfg = load_config("configs/concat_fusion.yaml")
    cfg["data"]["image_size"] = 32
    cfg["data"]["patch_size"] = 8
    cfg["model"]["vision_embed_dim"] = 16
    cfg["model"]["fusion_hidden_dim"] = 16
    cfg["model"]["projector_hidden_dim"] = 16
    cfg["model"]["llm_embed_dim"] = 16

    ds = GeoSARDataset(
        processed_dir=synthetic_processed_dir,
        annotations_path=f"{synthetic_processed_dir}/annotations.json",
        split="train", image_size=32,
    )
    tokenizer = SimpleTokenizer.build_from_texts([s["caption"] for s in ds.samples])

    for ftype in ["concat", "gated", "cross_attention", "none_eo", "none_sar"]:
        cfg["model"]["fusion_type"] = ftype
        model = GeoSARVLM(cfg, vocab_size=len(tokenizer))
        sample = ds[0]
        batch = {"sar": sample["sar"].unsqueeze(0), "eo": sample["eo"].unsqueeze(0)}
        text_ids = torch.tensor([tokenizer.encode(sample["caption"])])
        logits = model(batch, text_ids)
        assert logits.shape[0] == 1
        assert logits.shape[-1] == len(tokenizer)


@requires_torch
def test_retrieval_head_and_topk():
    import torch

    from src.models.retrieval import RetrievalHead, contrastive_loss, topk_retrieval

    head = RetrievalHead(vision_dim=16, text_dim=16, shared_dim=8)
    fused = torch.randn(6, 16)
    text = torch.randn(6, 16)
    logits_per_image, logits_per_text = head(fused, text)
    assert logits_per_image.shape == (6, 6)
    loss = contrastive_loss(logits_per_image)
    assert loss.item() > 0

    img_emb = head.encode_image(fused)
    txt_emb = head.encode_text(text)
    idx = topk_retrieval(txt_emb, img_emb, k=3)
    assert idx.shape == (6, 3)
