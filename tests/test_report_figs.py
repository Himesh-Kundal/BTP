"""Unit test for report figures and summary generation (T7.6)."""

from scripts.generate_report_figs import (
    FIG_DIR,
    RESULTS_DIR,
    generate_synthesis_summary_report,
    plot_fig1_area_scaling,
    plot_fig2_weight_distribution,
    plot_fig3_critical_path_depth,
    plot_fig4_full_layer_extrapolation,
    setup_plot_style,
)


def test_generate_report_figs_artifacts() -> None:
    """Verify that all figures and summary markdown are created and nonempty."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    plot_fig1_area_scaling()
    plot_fig2_weight_distribution()
    plot_fig3_critical_path_depth()
    plot_fig4_full_layer_extrapolation()
    generate_synthesis_summary_report()

    for fig_id in [1, 2, 3, 4]:
        matches_png = list(FIG_DIR.glob(f"fig{fig_id}_*.png"))
        matches_svg = list(FIG_DIR.glob(f"fig{fig_id}_*.svg"))
        assert len(matches_png) == 1, f"Missing PNG for figure {fig_id}"
        assert len(matches_svg) == 1, f"Missing SVG for figure {fig_id}"
        assert matches_png[0].stat().st_size > 1000
        assert matches_svg[0].stat().st_size > 1000

    summary_file = RESULTS_DIR / "synthesis_summary.md"
    assert summary_file.exists()
    assert summary_file.stat().st_size > 500
    content = summary_file.read_text(encoding="utf-8")
    assert "Executive Summary" in content
    assert "Pre-Layout Logic Synthesis" in content
