# lm-text-sanitization

Zero-Shot Redaction & Substitution pipeline with Large Language Models.

Paper: Albanese, F; Ciolek, D.; D’Ippolito, N. [Text Sanitization Beyond Specific Domains: Zero-Shot Redaction & Substitution with Large Language Models](https://arxiv.org/pdf/2311.10785.pdf).

[presentation](https://ppai-workshop.github.io/#program) at the PPAI @ AAAI 2024

## Abstract:

In the context of information systems, text sanitization techniques are used to identify and remove sensitive data to comply with security and regulatory requirements. Even though many methods for privacy preservation have been proposed, most of them are focused on the detection of entities from specific domains (e.g., credit card numbers, social security numbers), lacking generality and requiring customization for each desirable domain. 

Moreover, removing words is, in general, a drastic measure, as it can degrade text coherence and contextual information. Less severe measures include substituting a word for a safe alternative, yet it can be challenging to automatically find meaningful substitutions. We present a zero-shot text sanitization technique that detects and substitutes potentially sensitive information using Large Language Models. Our evaluation shows that our method excels at protecting privacy while maintaining text coherence and contextual information, preserving data utility for downstream tasks.

## Cite:

If you want to use some of these ideas or code in your own work, you can cite our paper on Zero-Shot Redaction & Substitution:
```
@article{albanese2023text,
  title={Text Sanitization Beyond Specific Domains: Zero-Shot Redaction \& Substitution with Large Language Models},
  author={Albanese, Federico and Ciolek, Daniel and D'Ippolito, Nicolas},
  journal={arXiv preprint arXiv:2311.10785},
  year={2023}
}
```
