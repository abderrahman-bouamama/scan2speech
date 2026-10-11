#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multilingual Text Refinement & Diacritization Engine

Author: Abderrahmane Bouamama
Project: Scan2Speech

Applies precise linguistic enhancement, diacritization, and TTS polishing using
any OpenAI-compatible Chat Completion API (DeepSeek, OpenAI, Groq, Ollama, OpenRouter, etc.).
- Arabic: Classical Tashkeel (Harakat) for flawless classical speech recitation
- English: Spoken cadence, punctuation refinement, phonetic normalization
- French: Accent restoration (é, è, à, ç), liaisons, natural audiobook pacing
- Spanish: Orthographic accents (tildes), inverted punctuation, spoken flow
- German: Umlauts (ä, ö, ü, ß), compound word clarity, narration cadence
- Italian: Accent marks (à, è, é), elisions, rhythmic audiobook pacing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# 1. MULTILINGUAL SYSTEM PROMPTS FOR TTS POLISHING & DIACRITIZATION
# ==============================================================================
# Below are 6 specialized system prompts for the most used languages.
# You can uncomment, modify, or select your preferred prompt via --language.

# ------------------------------------------------------------------------------
# PROMPT 1: ARABIC (العربية) — Classical Tashkeel (Diacritization)
# ------------------------------------------------------------------------------
# PROMPT_ARABIC = (
#     "أنت خبير لغوي متخصص في النحو والصرف والقراءة الإذاعية العربية الفصحى.\n"
#     "مهمتك: تشكيل النص العربي التالي تشكيلاً كاملاً ودقيقاً بالحركات (فتحة، ضمة، كسرة، سكون، شدة، تنوين) "
#     "مع مراعاة القواعد النحوية والصرفية الدقيقة ليكون جاهزاً لإلقاء وثائقي واحترافي متصل الفصاحة.\n"
#     "تعليمات حاسمة:\n"
#     "1. حافظ على نفس صياغة النص تماماً دون زيادة كلمة واحدة أو حذفها.\n"
#     "2. حافظ على علامات الترقيم (الفواصل، النقاط، علامات الاستفهام).\n"
#     "3. لا تضع نقاطاً متكررة (..) بل استعمل الفاصلة (،) أو النقطة (.).\n"
#     "4. لا تكتب أي مقدمة أو خاتمة أو تعليق أو علامات ماركداون، فقط النص المشكول مباشرة."
# )

# ------------------------------------------------------------------------------
# PROMPT 2: ENGLISH — Audiobook & Spoken Narration Polishing
# ------------------------------------------------------------------------------
# PROMPT_ENGLISH = (
#     "You are an expert audio editor and voiceover script specialist.\n"
#     "Your task: Polish and normalize the following English text for seamless, professional "
#     "Text-to-Speech (TTS) audiobook narration.\n"
#     "Critical instructions:\n"
#     "1. Preserve the exact meaning and original author vocabulary without altering the style.\n"
#     "2. Fix broken hyphenated line wraps and clear OCR artifacts.\n"
#     "3. Ensure clean punctuation (use standard commas, semicolons, and periods to guide natural speech pauses).\n"
#     "4. Replace repetitive ellipses or broken dashes with appropriate pauses (commas or em-dashes).\n"
#     "5. Output ONLY the polished body text directly. No preamble, notes, or markdown formatting."
# )

# ------------------------------------------------------------------------------
# PROMPT 3: FRENCH (Français) — Restauration des Accents et Ponctuation
# ------------------------------------------------------------------------------
# PROMPT_FRENCH = (
#     "Tu es un linguiste et réviseur expert en littérature et diction radiophonique française.\n"
#     "Ta mission: Réviser et préparer le texte français suivant pour une synthèse vocale (TTS) naturelle et élégante.\n"
#     "Instructions strictes:\n"
#     "1. Conserve scrupuleusement le texte d'origine sans ajouter ni supprimer de mots.\n"
#     "2. Restaure tous les accents manquants (é, è, ê, ë, à, â, î, ï, ô, ù, û, ç) issus d'erreurs OCR.\n"
#     "3. Harmonise la ponctuation pour fluidifier le rythme de lecture audio.\n"
#     "4. Évite les points de suspension multiples inutiles; privilégie la virgule ou le point.\n"
#     "5. Renvoie UNIQUEMENT le texte corrigé, sans introduction, sans commentaire et sans balises markdown."
# )

# ------------------------------------------------------------------------------
# PROMPT 4: SPANISH (Español) — Acentuación y Cadencia Narrativa
# ------------------------------------------------------------------------------
# PROMPT_SPANISH = (
#     "Eres un filólogo y corrector de estilo especializado en locución y audiolibros en español.\n"
#     "Tu tarea: Pulir y preparar el siguiente texto para una narración de voz sintetizada (TTS) fluida y natural.\n"
#     "Instrucciones obligatorias:\n"
#     "1. Mantén intacta la redacción original sin omitir ni añadir palabras.\n"
#     "2. Restaura todas las tildes diacríticas y ortográficas faltantes (á, é, í, ó, ú, ñ).\n"
#     "3. Asegura la correcta puntuación en español (incluyendo signos de apertura ¿?, ¡! si aplica).\n"
#     "4. Elimina guiones rotos de fin de línea y normaliza pausas de lectura.\n"
#     "5. Devuelve ÚNICAMENTE el texto optimizado, sin introducciones ni comentarios explicativos."
# )

# ------------------------------------------------------------------------------
# PROMPT 5: GERMAN (Deutsch) — Umlaut-Korrektur und Sprechrhythmus
# ------------------------------------------------------------------------------
# PROMPT_GERMAN = (
#     "Du bist ein erfahrener Lektor und Sprachspezialist für Hörbuch- und Rundfunkproduktion im Deutschen.\n"
#     "Deine Aufgabe: Den folgenden deutschen Text für eine natürliche und flüssige Text-to-Speech (TTS) Vertonung aufzubereiten.\n"
#     "Wichtige Richtlinien:\n"
#     "1. Behalte den exakten Wortlaut und Stil des Originaltextes bei.\n"
#     "2. Stelle alle fehlenden Umlaute (ä, ö, ü, Ä, Ö, Ü) und das Eszett (ß) wieder her.\n"
#     "3. Bereinige fehlerhafte OCR-Trennungen und normalisiere Satzzeichen für natürliche Sprechpausen.\n"
#     "4. Gib AUSSCHLIESSLICH den bereinigten Text aus, ohne Einleitung, Anmerkungen oder Markdown-Formatierung."
# )

# ------------------------------------------------------------------------------
# PROMPT 6: ITALIAN (Italiano) — Accentuazione e Scorrevolezza per Audiolibri
# ------------------------------------------------------------------------------
# PROMPT_ITALIAN = (
#     "Sei un linguista e fonetista esperto nella produzione di audiolibri e narrazioni orali in lingua italiana.\n"
#     "Il tuo compito: Perfezionare il seguente testo per una lettura Text-to-Speech (TTS) naturale e scorrevole.\n"
#     "Istruzioni fondamentali:\n"
#     "1. Mantieni fedelmente il testo originale senza modificare il significato o aggiungere vocaboli.\n"
#     "2. Ripristina tutti gli accenti grafici corretti (à, è, é, ì, ò, ù) ed elisioni (apostrofi).\n"
#     "3. Elimina trattini di fine riga errati e standardizza la punteggiatura per guidare le pause respiratorie.\n"
#     "4. Restituisci ESCLUSIVAMENTE il testo elaborato, senza introduzioni, commenti o blocchi markdown."
# )

# Active dictionary mapping language codes to their respective system prompts
SYSTEM_PROMPTS: dict[str, str] = {
    "ar": (
        "أنت خبير لغوي متخصص في النحو والصرف والقراءة الإذاعية العربية الفصحى.\n"
        "مهمتك: تشكيل النص العربي التالي تشكيلاً كاملاً ودقيقاً بالحركات (فتحة، ضمة، كسرة، سكون، شدة، تنوين) "
        "مع مراعاة القواعد النحوية والصرفية الدقيقة ليكون جاهزاً لإلقاء وثائقي واحترافي متصل الفصاحة.\n"
        "تعليمات حاسمة:\n"
        "1. حافظ على نفس صياغة النص تماماً دون زيادة كلمة واحدة أو حذفها.\n"
        "2. حافظ على علامات الترقيم (الفواصل، النقاط، علامات الاستفهام).\n"
        "3. لا تضع نقاطاً متكررة (..) بل استعمل الفاصلة (،) أو النقطة (.).\n"
        "4. لا تكتب أي مقدمة أو خاتمة أو تعليق أو علامات ماركداون، فقط النص المشكول مباشرة."
    ),
    "en": (
        "You are an expert audio editor and voiceover script specialist.\n"
        "Your task: Polish and normalize the following English text for seamless, professional "
        "Text-to-Speech (TTS) audiobook narration.\n"
        "Critical instructions:\n"
        "1. Preserve the exact meaning and original author vocabulary without altering the style.\n"
        "2. Fix broken hyphenated line wraps and clear OCR artifacts.\n"
        "3. Ensure clean punctuation (use standard commas, semicolons, and periods to guide natural speech pauses).\n"
        "4. Replace repetitive ellipses or broken dashes with appropriate pauses (commas or em-dashes).\n"
        "5. Output ONLY the polished body text directly. No preamble, notes, or markdown formatting."
    ),
    "fr": (
        "Tu es un linguiste et réviseur expert en littérature et diction radiophonique française.\n"
        "Ta mission: Réviser et préparer le texte français suivant pour une synthèse vocale (TTS) naturelle et élégante.\n"
        "Instructions strictes:\n"
        "1. Conserve scrupuleusement le texte d'origine sans ajouter ni supprimer de mots.\n"
        "2. Restaure tous les accents manquants (é, è, ê, ë, à, â, î, ï, ô, ù, û, ç) issus d'erreurs OCR.\n"
        "3. Harmonise la ponctuation pour fluidifier le rythme de lecture audio.\n"
        "4. Évite les points de suspension multiples inutiles; privilégie la virgule ou le point.\n"
        "5. Renvoie UNIQUEMENT le texte corrigé, sans introduction, sans commentaire et sans balises markdown."
    ),
    "es": (
        "Eres un filólogo y corrector de estilo especializado en locución y audiolibros en español.\n"
        "Tu tarea: Pulir y preparar el siguiente texto para una narración de voz sintetizada (TTS) fluida y natural.\n"
        "Instrucciones obligatorias:\n"
        "1. Mantén intacta la redacción original sin omitir ni añadir palabras.\n"
        "2. Restaura todas las tildes diacríticas y ortográficas faltantes (á, é, í, ó, ú, ñ).\n"
        "3. Asegura la correcta puntuación en español (incluyendo signos de apertura ¿?, ¡! si aplica).\n"
        "4. Elimina guiones rotos de fin de línea y normaliza pausas de lectura.\n"
        "5. Devuelve ÚNICAMENTE el texto optimizado, sin introducciones ni comentarios explicativos."
    ),
    "de": (
        "Du bist ein erfahrener Lektor und Sprachspezialist für Hörbuch- und Rundfunkproduktion im Deutschen.\n"
        "Deine Aufgabe: Den folgenden deutschen Text für eine natürliche und flüssige Text-to-Speech (TTS) Vertonung aufzubereiten.\n"
        "Wichtige Richtlinien:\n"
        "1. Behalte den exakten Wortlaut und Stil des Originaltextes bei.\n"
        "2. Stelle alle fehlenden Umlaute (ä, ö, ü, Ä, Ö, Ü) und das Eszett (ß) wieder her.\n"
        "3. Bereinige fehlerhafte OCR-Trennungen und normalisiere Satzzeichen für natürliche Sprechpausen.\n"
        "4. Gib AUSSCHLIESSLICH den bereinigten Text aus, ohne Einleitung, Anmerkungen oder Markdown-Formatierung."
    ),
    "it": (
        "Sei un linguista e fonetista esperto nella produzione di audiolibri e narrazioni orali in lingua italiana.\n"
        "Il tuo compito: Perfezionare il seguente testo per una lettura Text-to-Speech (TTS) naturale e scorrevole.\n"
        "Istruzioni fondamentali:\n"
        "1. Mantieni fedelmente il testo originale senza modificare il significato o aggiungere vocaboli.\n"
        "2. Ripristina tutti gli accenti grafici corretti (à, è, é, ì, ò, ù) ed elisioni (apostrofi).\n"
        "3. Elimina trattini di fine riga errati e standardizza la punteggiatura per guidare le pause respiratorie.\n"
        "4. Restituisci ESCLUSIVAMENTE il testo elaborato, senza introduzioni, commenti o blocchi markdown."
    ),
}


# ==============================================================================
# 2. API ENDPOINT CONFIGURATION (LINE 73 / USER CONFIGURABLE)
# ==============================================================================
# You can use ANY OpenAI-compatible chat completions API endpoint here!
# Set it via .env (LLM_API_URL or DEEPSEEK_API_URL), pass --api-url, or change
# DEFAULT_API_URL directly below.
#
# Examples of supported providers:
# - DeepSeek:    "https://api.deepseek.com/v1/chat/completions"
# - OpenAI:      "https://api.openai.com/v1/chat/completions"
# - Groq:        "https://api.groq.com/openai/v1/chat/completions"
# - OpenRouter:  "https://openrouter.ai/api/v1/chat/completions"
# - Local/Ollama:"http://localhost:11434/v1/chat/completions"
# - vLLM/LocalAI:"http://localhost:8000/v1/chat/completions"
DEFAULT_API_URL = (
    os.getenv("LLM_API_URL")
    or os.getenv("DEEPSEEK_API_URL")
    or "https://api.deepseek.com/v1/chat/completions"
)


def diacritize_chunk(
    text_chunk: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
    language: str = "ar",
    retry_count: int = 3,
) -> str:
    """Send a text block to the LLM completion API for linguistic refinement."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    active_prompt = system_prompt or SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["ar"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": active_prompt},
            {"role": "user", "content": text_chunk},
        ],
        "temperature": 0.15,
    }

    for attempt in range(retry_count):
        try:
            # Line 73: Dynamic OpenAI-compatible endpoint (DeepSeek, OpenAI, Groq, Ollama, etc.)
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```.*?\n", "", content)
                content = re.sub(r"\n```$", "", content)
                return content.strip()
            else:
                print(f"    [WARN] API HTTP {resp.status_code}: {resp.text[:120]} (attempt {attempt + 1})")
                time.sleep(2)
        except Exception as e:
            print(f"    [WARN] Connection error to {api_url}: {e} (attempt {attempt + 1})")
            time.sleep(3)

    raise RuntimeError(f"Failed to process chunk after {retry_count} attempts using endpoint: {api_url}")


def process_text_content(
    raw_text: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    model: str = "deepseek-chat",
    language: str = "ar",
    system_prompt: str | None = None,
    chunk_size: int = 1100,
) -> str:
    """Break text into safe chunks and process sequentially."""
    lines = [
        re.sub(r"[ \t]+", " ", line.strip())
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("===")
    ]

    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        current_chunk.append(line)
        current_len += len(line)
        if current_len >= chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    diacritized_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"    -> Chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
        diacritized = diacritize_chunk(
            text_chunk=chunk,
            api_key=api_key,
            api_url=api_url,
            model=model,
            system_prompt=system_prompt,
            language=language,
        )
        diacritized_parts.append(diacritized)
        time.sleep(0.4)

    full_diacritized_lines = []
    for part in diacritized_parts:
        for line in part.splitlines():
            l_clean = line.strip()
            if l_clean and not l_clean.startswith("==="):
                full_diacritized_lines.append(l_clean)

    return "\n".join(full_diacritized_lines) + "\n"


def diacritize_file(
    input_file: Path,
    output_file: Path,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    model: str = "deepseek-chat",
    language: str = "ar",
    system_prompt: str | None = None,
    chunk_size: int = 1100,
    resume: bool = True,
) -> None:
    """Refine / diacritize a single file."""
    if resume and output_file.exists() and output_file.stat().st_size > 500:
        print(f"  [CACHE] File already processed: {output_file.name}")
        return

    print(f"\n[RUN] Processing ({language.upper()}): {input_file.name}...")
    raw = input_file.read_text(encoding="utf-8")
    result = process_text_content(
        raw_text=raw,
        api_key=api_key,
        api_url=api_url,
        model=model,
        language=language,
        system_prompt=system_prompt,
        chunk_size=chunk_size,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result, encoding="utf-8")
    print(f"[OK] Saved processed text: {output_file}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Multilingual Text Diacritization & TTS Polishing Engine. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        default=None,
        help="Path to a single text file to process",
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        type=Path,
        default=None,
        help="Directory containing chapter text files",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Destination path for single output file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for processed chapter files",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["ar", "en", "fr", "es", "de", "it"],
        default="ar",
        help="Target language: ar (Arabic Tashkeel), en (English), fr (French), es (Spanish), de (German), it (Italian). Default: ar",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (defaults to DEEPSEEK_API_KEY, OPENAI_API_KEY, or LLM_API_KEY from .env)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=f"Chat completions API URL (defaults to LLM_API_URL or DEFAULT_API_URL: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name (e.g. deepseek-chat, gpt-4o-mini, llama-3.3-70b-versatile. Default: deepseek-chat)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt override (overrides default language prompt)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1100,
        help="Max characters per chunk sent to API (default: 1100)",
    )

    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("LLM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    api_key = api_key.strip().strip('"').strip("'")

    if not api_key:
        print("[ERROR] API Key not found. Please provide --api-key or set DEEPSEEK_API_KEY / LLM_API_KEY in .env.")
        sys.exit(1)

    api_url = args.api_url or DEFAULT_API_URL

    if not args.input_file and not args.input_dir:
        print("[ERROR] Please specify either --input-file or --input-dir.")
        sys.exit(1)

    print("=" * 65)
    print("  Multilingual Text Diacritization & Polish Engine")
    print("  Creator  : Abderrahmane Bouamama")
    print(f"  Language : {args.language.upper()}")
    print(f"  API URL  : {api_url}")
    print(f"  Model    : {args.model}")
    print("=" * 65)

    if args.input_file:
        suffix = "_tashkeel.txt" if args.language == "ar" else f"_{args.language}_polished.txt"
        out_file = args.output_file or args.input_file.with_name(f"{args.input_file.stem}{suffix}")
        diacritize_file(
            input_file=args.input_file,
            output_file=out_file,
            api_key=api_key,
            api_url=api_url,
            model=args.model,
            language=args.language,
            system_prompt=args.system_prompt,
            chunk_size=args.chunk_size,
        )

    elif args.input_dir:
        suffix = "-tashkeel" if args.language == "ar" else f"-{args.language}-polished"
        out_dir = args.output_dir or args.input_dir.parent / f"{args.input_dir.name}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(args.input_dir.glob("*.txt"))
        if not txt_files:
            print(f"[ERROR] No .txt files found in {args.input_dir}")
            sys.exit(1)

        print(f"Found {len(txt_files)} files in {args.input_dir}...")
        for f in txt_files:
            if f.name.startswith("master_") or f.name.startswith("script_audiobook_complet"):
                continue
            out_f = out_dir / f.name
            diacritize_file(
                input_file=f,
                output_file=out_f,
                api_key=api_key,
                api_url=api_url,
                model=args.model,
                language=args.language,
                system_prompt=args.system_prompt,
                chunk_size=args.chunk_size,
            )

        # Merge processed files into a single master file
        all_processed = []
        for f in sorted(out_dir.glob("*.txt")):
            if f.name.startswith("master_"):
                continue
            all_processed.append(f.read_text(encoding="utf-8").strip())

        master_out = out_dir / f"master_script_{args.language}.txt"
        master_out.write_text("\n\n".join(all_processed) + "\n", encoding="utf-8")
        print(f"\n[DONE] Master processed script saved: {master_out}")


if __name__ == "__main__":
    main()
