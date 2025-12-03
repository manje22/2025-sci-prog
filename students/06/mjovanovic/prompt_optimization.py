import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ------------------------------------------------------------
# Load API key
# ------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)
smooth = SmoothingFunction().method4

# ------------------------------------------------------------
# Dataset kao jedan duži tekst
# ------------------------------------------------------------
text_to_translate = """
Funkcija vraća rezultat izračuna.
Sustav prikazuje poruku o pogrešci.
Modul je odgovoran za obradu podataka.
Datoteka se ne može otvoriti zbog nedostatnih dopuštenja.
Aplikacija koristi API za pristup vanjskim servisima.
Parametri funkcije moraju biti definirani prije poziva metode.
Server je nedostupan zbog greške u mrežnoj konfiguraciji.
Baza podataka podržava transakcije i referencijalni integritet.
Kod koristi rekurzivnu funkciju za obradu stabla podataka.
Modul za autentikaciju provjerava korisničke pristupne podatke.
Sustav generira logove za praćenje pogrešaka.
Klasa implementira sučelje za upravljanje uređajima.
Algoritam sortira podatke po abecednom redu.
Paket je instaliran pomoću upravitelja paketa.
Metoda baca iznimku ako je ulaz neispravan.
Aplikacija koristi višedretvene procese za poboljšanje performansi.
Konfiguracijska datoteka sadrži postavke sustava.
Funkcija dohvaća podatke s udaljenog poslužitelja.
Program zapisuje rezultate u CSV datoteku.
Upravljački modul inicijalizira sve servise prilikom pokretanja.
Sustav koristi enkripciju za zaštitu osjetljivih podataka.
Modul za mrežnu komunikaciju koristi TCP protokol.
Aplikacija podržava višestruke korisničke sesije.
Kod uključuje komentare za bolju čitljivost.
API vraća JSON objekt sa statusnim kodom.
Funkcija provjerava ispravnost unesenih podataka.
Modul za logiranje zapisuje sve akcije korisnika.
Sustav koristi predmemoriju za brži pristup podacima.
Metoda koristi rekurziju za obradu hijerarhijskih struktura.
Aplikacija učitava konfiguraciju prilikom pokretanja.
Klasa nasljeđuje osnovnu funkcionalnost iz roditeljske klase.
Program provjerava kompatibilnost verzija prije instalacije.
Funkcija šalje zahtjev na vanjski servis i čeka odgovor.
Sustav generira upozorenja za neuobičajene aktivnosti.
Datoteka sadrži binarne podatke koji zahtijevaju poseban parser.
API omogućuje autentikaciju putem tokena.
Modul za obradu slike koristi filtere za poboljšanje kvalitete.
Funkcija paralelno obrađuje podatke pomoću više niti.
Program koristi algoritam pretraživanja za pronalazak rješenja.
Sustav provjerava integritet podataka prilikom prijenosa.
Modul za sigurnost šifrira i dešifrira poruke.
"""

reference_lines = [line.strip() for line in text_to_translate.strip().split("\n")]

# ------------------------------------------------------------
# Initial prompt
# ------------------------------------------------------------
initial_prompt = """
You are an expert Croatian-to-English translator.
Translate highly technical AI/ML/cybersecurity documents.
Preserve terminology (e.g., "funkcija"="function", "model"="model", "skup podataka"="dataset", "preuzorkovanje"="resampling").
Maintain BLEU-like precision and sentence fidelity.
"""

# ------------------------------------------------------------
# Helper functions s retry
# ------------------------------------------------------------
def run_flash_lite(prompt, text):
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                config=types.GenerateContentConfig(system_instruction=prompt),
                contents=text
            )
            return response.text.strip()
        except ClientError as e:
            if e.status_code == 429:
                wait_time = float(e.args[0].split("Please retry in ")[1].split("s")[0])
                print(f"Quota exceeded. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 1)
            else:
                raise e
        except ServerError:
            wait_time = (attempt + 1) * 5
            print(f"Model overloaded. Retry in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Flash-Lite request failed after multiple retries.")

def run_gemini_pro(prompt, input_text, output_text):
    judge_input = f"""
You are a prompt optimization judge.

Task: Translate Croatian technical text to English.
Current Flash-Lite translation:

{output_text}

Your job:
- Identify errors
- Produce a BETTER system prompt

Respond with format:

ERRORS:
- bullet points

NEW_PROMPT:
<<<
(new prompt here)
>>>

REASONING:
(one paragraph)
"""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=judge_input
            )
            return response.text
        except (ClientError, ServerError):
            wait_time = (attempt + 1) * 5
            print(f"Judge busy. Retry in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Gemini Pro judge request failed after multiple retries.")

def extract_new_prompt(judge_response):
    pattern = r"NEW_PROMPT:\s*<<<(.*?)>>>"
    match = re.search(pattern, judge_response, re.DOTALL)
    if not match:
        raise ValueError("Judge did not return NEW_PROMPT block")
    return match.group(1).strip()

def compute_bleu(references, hypothesis_text):
    hyp_lines = [line.strip() for line in hypothesis_text.strip().split("\n")]
    bleu_scores = []
    for ref, hyp in zip(references, hyp_lines):
        bleu_scores.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth))
    return sum(bleu_scores) / len(bleu_scores)

# ------------------------------------------------------------
# Main iterative optimizer + CLI loop
# ------------------------------------------------------------
def main():
    MAX_ITERATIONS = 5
    current_prompt = initial_prompt

    # ---------- Prompt optimization ----------
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n================ Iteration {iteration} ================")

        translation = run_flash_lite(current_prompt, text_to_translate)
        print("\nFlash-Lite translation preview:\n", translation[:500], "...")

        avg_bleu = compute_bleu(reference_lines, translation)
        print(f"\nAverage BLEU on dataset: {avg_bleu:.4f}")

        judge_output = run_gemini_pro(current_prompt, text_to_translate, translation)
        print("\nJudge output preview:\n", judge_output[:500], "...")

        try:
            new_prompt = extract_new_prompt(judge_output)
        except ValueError:
            print("Could not extract new prompt. Stopping iterations.")
            break

        if new_prompt == current_prompt:
            print("Prompt did not change. Converged.")
            break

        current_prompt = new_prompt
        print("\nNew optimized prompt preview:\n", current_prompt[:500], "...")

    print("\n================ Final Optimized Prompt ================")
    print(current_prompt)

    # ---------- CLI loop ----------
    print("\nPrompt optimiziran. Unesite tekst za prijevod ili 'exit' za izlaz.\n")
    while True:
        user_input = input("Unesite tekst za prijevod: ").strip()
        if user_input.lower() == "exit":
            print("Izlaz iz programa. Doviđenja!")
            break
        elif user_input == "":
            continue

        try:
            translation = run_flash_lite(current_prompt, user_input)
            print("\nPrijevod:\n", translation, "\n")
        except Exception as e:
            print(f"Greška pri prijevodu: {e}")

if __name__ == "__main__":
    main()
