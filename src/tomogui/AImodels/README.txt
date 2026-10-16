Place AI model weight files (.pt / .pth) here.

Two DIFFERENT model checkpoints are consumed by the AI COR search:

    fine_v2.pt   -- FINE model (--infer-model-path)
                    Used by the single-stage inference_pipeline
                    (ai-search-method = fine), and also as the
                    final stage of the two-stage full search.
                    Default location:
                      <tomoguiAI env>/site-packages/tomogui/AImodels/fine_v2.pt

    epoch_26.pt  -- FULL model (--bin-infer-model-path)
                    Used by the two-stage bin_inference_pipeline
                    (ai-search-method = full). The two-stage search
                    runs the FULL (coarse) stage first, then falls
                    back to the FINE model for the last refinement.
                    Current default (as of 2026-10-13):
                      /home/beams/TANGS/conda/tomocor_models/
                        2026_09_24-11_37_44-model_ViT-L-16-lr_2e-05-b_2-j_4-p_amp/
                        checkpoints/epoch_26.pt

They are NOT interchangeable — the FULL checkpoint includes the extra
feature-aggregator layers (see --bin-infer-aggregator-depth /
--bin-infer-aggregator-num-heads). Loading the FINE checkpoint as the
FULL model (or vice versa) will crash out of the coarse stage before
the FINE stage ever runs.

The GUI's "AI COR" tab defaults both path fields to the files above.
Both can be overridden per-run via the Browse button.

Model files are large (> 1 GB each) and are NOT stored in git.
Download / copy them into their respective locations before running AI
mode.
