"""Public demo: paste a ticket in any language, get a category and a priority.

Runs on CPU. Deploy to Hugging Face Spaces (free tier is enough) or anywhere
that can run a Python process:

    pip install -e ".[demo]"
    ticket-triage train              # writes artifacts/model.joblib
    python app.py

The banner states what the model was trained on, because a demo that implies
more than it can deliver costs more credibility than it earns.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from ticket_triage import TicketTriageModel

MODEL_PATH = Path(os.environ.get("TICKET_TRIAGE_MODEL", "artifacts/model.joblib"))

EXAMPLES = [
    "Impossible de se connecter au VPN depuis ce matin, erreur après quelques secondes",
    "l vpn kat9ta3 kol chi 10 d daqayq, 3afak chofo lia had chi dghya",
    "تظهر الطابعة رسالة انحشار الورق رغم أن الدرج فارغ",
    "J'ai reçu un mail demandant mes identifiants avec un lien externe, je n'ai pas cliqué",
    "nssit l mot de passe dyali o l compte tsedd",
    "صندوق البريد ممتلئ ولم أعد أستقبل أي رسالة منذ الأمس",
]

DISCLAIMER = """
### What you are looking at

A multilingual ITSM ticket classifier — **French, Modern Standard Arabic and
Darija** — running on CPU. No GPU, no external API, 0.5 MB model.

**It was trained entirely on synthetic tickets.** No customer ticket was used at
any point, which is why it can be published at all. What the measurements
actually say:

| | |
|---|---|
| **Category: 1.000** | Not an achievement. The seven categories use disjoint vocabulary by construction, so a linear model separates them perfectly. Read it as "the pipeline works". |
| **Priority: 0.901** | The meaningful figure — and it is measured against the **0.513** reachable from the topic alone. The 0.39 on top comes from reading how many people are affected and how soon it matters. |
| **Across languages** | 0.954 Arabic / 0.894 Darija / 0.887 French on priority. **This is the result that matters**: the model does not favour one language. |

Numbers on your own tickets will be lower — the test set is synthetic, and a test
ticket is a template sibling of a training one. Getting a figure that means
something for your service desk means fine-tuning on your data, which is a
scoping exercise rather than a download.
"""


def load_model() -> TicketTriageModel:
    if not MODEL_PATH.exists():
        raise SystemExit(f"no model at {MODEL_PATH}. Run: ticket-triage train")
    return TicketTriageModel.load(MODEL_PATH)


def build_interface(model: TicketTriageModel) -> gr.Blocks:
    def classify(text: str) -> tuple[dict[str, float], str, str]:
        if not text or not text.strip():
            return {}, "—", "Paste a ticket above."

        result = model.predict(text)
        categories = dict(model.top_categories(text, k=5))
        priority = (
            f"## P{result['priority']}\nconfidence {result['priority_confidence']:.0%}"
        )
        detail = (
            f"**{result['category']}** at {result['category_confidence']:.0%} "
            f"confidence.\n\nLow confidence means the ticket does not look like "
            f"anything in the catalogue — which is useful information in itself."
        )
        return categories, priority, detail

    with gr.Blocks(title="Multilingual ticket triage") as demo:
        gr.Markdown("# Multilingual ITSM ticket triage")
        gr.Markdown(
            "Paste a helpdesk ticket in **French, Arabic or Darija** "
            "(Arabic script or Latin letters)."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Ticket",
                    lines=5,
                    placeholder="l vpn kat9ta3 kol chi 10 d daqayq...",
                )
                submit = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                priority_out = gr.Markdown(label="Priority")
                categories_out = gr.Label(label="Category", num_top_classes=5)

        detail_out = gr.Markdown()
        gr.Examples(examples=[[example] for example in EXAMPLES], inputs=text)
        gr.Markdown(DISCLAIMER)

        submit.click(
            classify, inputs=text, outputs=[categories_out, priority_out, detail_out]
        )
        text.submit(
            classify, inputs=text, outputs=[categories_out, priority_out, detail_out]
        )

    return demo


if __name__ == "__main__":
    build_interface(load_model()).launch()
