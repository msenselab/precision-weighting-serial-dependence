# Figures

This directory contains the canonical manuscript and appendix figure exports.
These files are the authoritative final manuscript assets.

Included main figures: Fig. 1–7. Included appendix figures: Fig. A1 and Fig. B1–B2.

The plotting scripts in `code/` regenerate these exports in temporary staging
directories. They replace tracked PNG/PDF pairs only when the rendered PNG
dimensions or pixels change, avoiding churn from PNG compression and PDF
metadata alone.
