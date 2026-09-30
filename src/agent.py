"""FinAjan OpenAI Uyumlu Tool-Use Ajanı.

Kullanıcının doğal dildeki finansal sorularını analiz araçlarını kullanarak
otonom bir şekilde çözen döngü ve halüsinasyon koruma katmanı.
"""

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

import pandas as pd

from src.agent_tools import TOOL_DEFINITIONS, execute_tool, tool_get_dataset_info
from src.llm_client import (
    LLMClient,
    LLMConfigError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from src.prompts import GUARD_RETRY_TEMPLATE, build_system_prompt


@dataclass
class AgentResult:
    """Ajanın kullanıcı sorusuna verdiği nihai yanıtı ve meta verileri içerir."""
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    unverified_numbers: list[float] = field(default_factory=list)
    error: str | None = None


def extract_numbers_from_data(obj: Any) -> set[float]:
    """Sözlük, liste veya temel tiplerdeki sayısal değerleri özyinelemeli olarak toplar."""
    numbers: set[float] = set()
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        val = float(obj)
        numbers.add(round(val, 2))
        numbers.add(round(abs(val), 2))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers.update(extract_numbers_from_data(v))
    elif isinstance(obj, list):
        for item in obj:
            numbers.update(extract_numbers_from_data(item))
    return numbers


def extract_numbers_from_text(text: str) -> list[float]:
    """Cevap metnindeki sayıları (Türkçe 42.315,50 ve standart formatlar) çıkarır.
    
    Yıllar (2000-2099) ve <= 31 tam sayılar (günler, madde numaraları) elenir.
    Tek bir sayının alt parçalara bölünmesini engellemek için öncelikli tek regex kullanılır.
    """
    pattern = re.compile(
        r"\b(?:"
        r"(\d{1,3}(?:\.\d{3})+,\d+)|"  # 1. TR formatı: 42.315,50
        r"(\d{1,3}(?:,\d{3})+\.\d+)|"  # 2. US formatı: 42,315.50
        r"(\d{1,3}(?:\.\d{3})+)|"       # 3. TR binlik tam sayı: 42.315
        r"(\d{1,3}(?:,\d{3})+)|"       # 4. US binlik tam sayı: 42,315
        r"(\d+[,.]\d+)|"               # 5. Düz ondalık: 229,99 veya 229.99
        r"(\d+)"                       # 6. Düz tam sayı: 500
        r")\b"
    )

    candidates: list[float] = []
    seen: set[float] = set()

    for m in pattern.finditer(text):
        g1, g2, g3, g4, g5, g6 = m.groups()
        try:
            if g1:
                val = float(g1.replace(".", "").replace(",", "."))
            elif g2:
                val = float(g2.replace(",", ""))
            elif g3:
                val = float(g3.replace(".", ""))
            elif g4:
                val = float(g4.replace(",", ""))
            elif g5:
                val = float(g5.replace(",", "."))
            elif g6:
                val = float(g6)
            else:
                continue
        except ValueError:
            continue

        r_val = round(val, 2)
        if r_val in seen:
            continue
        seen.add(r_val)

        # Filtreleme kuralları:
        # 1. <= 31 tam sayılar (günler, maddeler) elenir
        # 2. 2000-2099 arası tam sayılar (yıllar) elenir
        is_int_like = (val == int(val))
        if is_int_like and val <= 31:
            continue
        if is_int_like and 2000 <= val <= 2099:
            continue

        candidates.append(r_val)

    return candidates


def is_number_verified(cand: float, known: set[float]) -> bool:
    """Aday sayının bilinen araç çıktıları kümesinde olup olmadığını toleransla teyit eder."""
    for k in known:
        # 1. Doğrudan eşitlik (0.05 tolerans)
        if abs(cand - k) < 0.05:
            return True
        # 2. Yüzde karşılığı (oran 0.0553 iken metinde %5.53 veya %5.5 yazılması)
        if abs(cand - (k * 100)) < 0.1:
            return True
        if abs((cand * 100) - k) < 0.1:
            return True
    return False


class FinanceAgent:
    """Kullanıcı finansal sorularını yöneten OpenAI Uyumlu Tool-Use Ajanı."""

    def __init__(self, llm: LLMClient, df: pd.DataFrame, max_steps: int = 6):
        """Ajan başlatıcı.
        
        Args:
            llm: LLMClient nesnesi.
            df: İşlem verilerini içeren Pandas DataFrame.
            max_steps: Bir soruda azami LLM düşünce/araç adımı sayısı (varsayılan: 6).
        """
        self.llm = llm
        self.df = df
        self.max_steps = max_steps
        # Turlar arası hafıza: Yalnızca düz metin (soru, cevap) çiftleri, son 10 tur
        self.history: list[dict[str, str]] = []
        # Ajan ömrü boyunca bilinen/doğrulanmış sayılar kümesi
        self.known_numbers: set[float] = set()

    def ask(self, question: str) -> AgentResult:
        """Kullanıcının sorusunu alır, araç döngüsünü çalıştırır ve doğrulanmış yanıtı döner.

        Args:
            question: Kullanıcının doğal dildeki sorusu.

        Returns:
            AgentResult: Yanıt metni, adımlar, doğrulanamayan sayılar ve hata durumu.
        """
        if not self.llm.is_available():
            return AgentResult(
                answer="LLM API anahtarı veya modeli tanımlanmamış. Lütfen .env dosyasında geçerli bir LLM_API_KEY ve LLM_MODEL belirtin.",
                steps=[],
                unverified_numbers=[],
                error="API anahtarı eksik",
            )

        # Veri seti meta bilgilerini topla ve sistem promptunu oluştur
        dataset_info = tool_get_dataset_info(self.df)
        system_prompt = build_system_prompt(dataset_info)

        # Veri seti temel rakamlarını bilinen sayılara ekle
        self.known_numbers.update(extract_numbers_from_data(dataset_info))

        # Turlar arası geçmişi yükle ve yeni soruyu ekle
        messages: list[dict[str, Any]] = []
        for h in self.history[-20:]:  # Son 10 tur (20 mesaj)
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})

        steps: list[dict[str, Any]] = []
        guard_retry_done = False
        final_answer = ""
        unverified_numbers: list[float] = []

        step_count = 0
        while step_count < self.max_steps:
            step_count += 1
            try:
                response = self.llm.create(
                    messages=messages,
                    system=system_prompt,
                    tools=TOOL_DEFINITIONS,
                )
            except (LLMConfigError, LLMConnectionError, LLMRateLimitError, LLMResponseError) as e:
                return AgentResult(
                    answer=f"Model çağrısı sırasında bir sorun oluştu: {str(e)}",
                    steps=steps,
                    unverified_numbers=[],
                    error=str(e),
                )
            except Exception as e:
                return AgentResult(
                    answer=f"Beklenmeyen bir hata oluştu: {type(e).__name__}",
                    steps=steps,
                    unverified_numbers=[],
                    error=str(e),
                )

            # 1. Durum: Model araç çağırmak istiyor (tool_calls mevcut)
            if response.tool_calls:
                # response.message'ı (assistant rolünde, tool_calls içeren temiz dict) messages'a ekle
                messages.append(response.message)

                # Her bir tool call için aracı çalıştır
                for tc in response.tool_calls:
                    t_name = tc["name"]
                    t_args = tc["arguments"]
                    t_id = tc["id"]

                    t_start = time.perf_counter()
                    if isinstance(t_args, dict) and t_args.get("_parse_error"):
                        output_dict = {"error": "Araç argümanları geçerli bir JSON formatında çözülemedi."}
                    else:
                        output_dict = execute_tool(t_name, t_args, self.df)
                    sure_ms = round((time.perf_counter() - t_start) * 1000, 1)

                    is_err = "error" in output_dict
                    # Çıktıdaki tüm sayıları doğrulanmış kümesine ekle
                    self.known_numbers.update(extract_numbers_from_data(output_dict))

                    output_ozeti = (
                        f"Hata: {output_dict['error']}"
                        if is_err
                        else f"Başarılı ({len(json.dumps(output_dict, ensure_ascii=False))} bayt)"
                    )

                    steps.append({
                        "tool": t_name,
                        "input": t_args,
                        "output_ozeti": output_ozeti,
                        "sure_ms": sure_ms,
                        "hata_mi": is_err,
                    })

                    # OpenAI protokolü: Her tool sonucu İÇİN AYRI BİR mesaj olarak eklenir
                    messages.append({
                        "role": "tool",
                        "tool_call_id": t_id,
                        "content": json.dumps(output_dict, ensure_ascii=False),
                    })

                continue

            # 2. Durum: Model metin döndürdü (tool_calls boş)
            candidate_text = response.text

            # Halüsinasyon kontrolü: Metindeki sayıları çıkar ve doğrula
            text_numbers = extract_numbers_from_text(candidate_text)
            unverified = [
                num for num in text_numbers
                if not is_number_verified(num, self.known_numbers)
            ]

            # Doğrulanamayan sayı varsa ve henüz retry yapılmadıysa: 1 kez uyar
            if unverified and not guard_retry_done:
                guard_retry_done = True
                unverified_str = ", ".join(f"{n:g}" for n in unverified)
                retry_instruction = GUARD_RETRY_TEMPLATE.format(unverified_list=unverified_str)

                # Asistan yanıtını ve retry user uyarısını ekleyip döngüye devam et
                messages.append({"role": "assistant", "content": response.text})
                messages.append({"role": "user", "content": retry_instruction})
                continue

            # Başarılı veya 2. deneme sonrası kabul
            final_answer = candidate_text
            unverified_numbers = unverified
            break

        # Azami adım aşıldıysa nazik bir mesaj dön
        if not final_answer:
            final_answer = (
                "İşleminiz çok fazla adım gerektirdiği için tamamlanamadı. "
                "Lütfen sorunuzu belirli bir tarih aralığı veya kategori belirterek tekrar sorunuz."
            )

        # Turlar arası hafızayı güncelle (yalnızca düz metin)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": final_answer})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return AgentResult(
            answer=final_answer,
            steps=steps,
            unverified_numbers=unverified_numbers,
            error=None,
        )
