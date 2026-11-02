# Detecting Forged Handwritten Signatures

UCS321 AI for Engineers, Assignment 2 and Report 1 (topic T-178).

A CNN on raw CEDAR scans scores 100%, but so does a one-line brightness rule, because genuine and forged images were scanned differently. With that shortcut removed and testing on signers the model has never seen, a reference-based CNN reaches 78.8% accuracy (ROC AUC 0.861).

| Folder | Contents |
|---|---|
| `notebook/` | Colab notebook, executed with all outputs |
| `report/` | Report 1 as PDF and Word, plus a shorter formal version in both formats |
| `results/` | Metrics and figures from the notebook run |
| `site/` | Static interactive dashboard, deployed on Vercel |
| `scripts/` | Build scripts for the notebook, report and dashboard data |

Data: CEDAR signature database (Kalera, Srihari and Xu, 2004).

Built with help from Claude (Anthropic).
