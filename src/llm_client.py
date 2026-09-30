"""FinAjan OpenAI Uyumlu LLM İstemci Modülü.

Model çağrı izolasyonu: Tüm LLM çağrıları yalnızca bu modül içinden yapılır.
Projenin başka hiçbir yerinde doğrudan OpenAI/Groq veya model çağrısı bulunamaz.
API anahtarı ve hassas finansal veriler asla loglanmaz veya dışa sızdırılmaz.
"""

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Sequence

from dotenv import load_dotenv
import openai

logger = logging.getLogger("finajan.llm_client")


# --- Özel Hata Sınıfları ---

class LLMError(Exception):
    """Temel LLM hata sınıfı."""
    pass


class LLMConfigError(LLMError):
    """API anahtarı veya model yapılandırması eksik/hatalı olduğunda fırlatılır."""
    pass


class LLMConnectionError(LLMError):
    """API sunucusuna ağ veya bağlantı kurulamadığında fırlatılır."""
    pass


class LLMRateLimitError(LLMError):
    """API kota veya hız sınırı (rate limit) aşıldığında fırlatılır."""
    pass


class LLMResponseError(LLMError):
    """API durumu, geçersiz yanıt veya beklenmeyen durma nedenlerinde fırlatılır."""
    pass


# --- Yanıt Veri Modeli ---

@dataclass
class LLMResponse:
    """LLM model yanıtını kapsülleyen veri sınıfı."""
    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    usage: dict[str, int]
    message: dict[str, Any]


# --- İstemci Sınıfı ---

class LLMClient:
    """OpenAI uyumlu API uç noktaları (Groq vb.) ile iletişim kuran lazy istemci."""

    VALID_FINISH_REASONS = {"stop", "tool_calls"}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        load_env: bool = True,
    ):
        """LLMClient başlatıcı.

        Args:
            api_key: LLM API anahtarı (belirtilmezse env'den okunur).
            base_url: API taban URL'i (varsayılan: https://api.groq.com/openai/v1).
            model: Model adı (varsayılan: openai/gpt-oss-120b).
            load_env: True ise .env dosyasını yükler; testlerde False verilerek .env izole edilir.
        """
        if load_env:
            load_dotenv()

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        
        env_model = os.getenv("LLM_MODEL")
        if model is not None:
            resolved_model = model
        elif env_model is not None:
            resolved_model = env_model
        else:
            resolved_model = "openai/gpt-oss-120b"

        if not resolved_model or not resolved_model.strip():
            raise LLMConfigError("LLM_MODEL tanımlı değil")
        self.model = resolved_model.strip()

        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT")
        self.timeout = 60.0
        self.max_retries = 2
        self._client: openai.OpenAI | None = None

    def is_available(self) -> bool:
        """API anahtarının ve modelin mevcut ve dolu olup olmadığını kontrol eder."""
        has_key = bool(self.api_key and self.api_key.strip())
        has_model = bool(self.model and self.model.strip())
        return has_key and has_model

    def _get_client(self) -> openai.OpenAI:
        """Tembel (lazy) OpenAI SDK istemcisi ilklendirici."""
        if self._client is None:
            if not self.is_available():
                raise LLMConfigError(
                    "LLM API anahtarı veya model tanımlı değil. Lütfen .env dosyasında LLM_API_KEY ve LLM_MODEL tanımlayın."
                )
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        """OpenAI uyumlu Chat Completions uç noktasına istek gönderir ve yanıtı ayrıştırır.

        Args:
            messages: Mesaj geçmişi listesi.
            system: Sistem istemi (prompt). Varsa mesajların en başına system rolüyle eklenir.
            tools: OpenAI function formatında araç tanımları listesi.
            max_tokens: Azami çıktı token sayısı (varsayılan: 1500).

        Returns:
            LLMResponse: Ayrıştırılmış metin, araç çağrıları, kullanım ve sonraki tura hazır assistant mesajı.

        Raises:
            LLMConfigError: API anahtarı veya model yoksa.
            LLMConnectionError: Ağ bağlantı hatası varsa.
            LLMRateLimitError: İstek veya kota sınırı aşılmışsa.
            LLMResponseError: Beklenmeyen durum veya bitiş nedenlerinde.
        """
        client = self._get_client()

        # Gelen mesaj listesini mutasyona uğratma; yeni liste oluştur
        req_messages: list[dict[str, Any]] = []
        if system:
            req_messages.append({"role": "system", "content": system})
        for msg in messages:
            req_messages.append(dict(msg))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": req_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        # Groq/OpenAI reasoning_effort parametresi (varsa extra_body ile iletilir)
        if self.reasoning_effort and self.reasoning_effort.strip():
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort.strip()}

        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM API rate limit aşıldı. Model: %s, Süre: %.1fms",
                self.model,
                duration_ms,
            )
            raise LLMRateLimitError(
                "Ücretsiz kullanım sınırına ulaşıldı, biraz bekleyip tekrar deneyin."
            )
        except openai.APIConnectionError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM API bağlantı hatası. Model: %s, Süre: %.1fms",
                self.model,
                duration_ms,
            )
            raise LLMConnectionError(
                "LLM API sunucusuna bağlanılamadı. Lütfen internet bağlantınızı kontrol edin."
            )
        except openai.APIStatusError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            status_code = getattr(e, "status_code", None)
            logger.error(
                "LLM API durum hatası: HTTP %s. Model: %s, Süre: %.1fms",
                status_code or "Bilinmiyor",
                self.model,
                duration_ms,
            )
            if status_code == 429:
                raise LLMRateLimitError(
                    "Ücretsiz kullanım sınırına ulaşıldı, biraz bekleyip tekrar deneyin."
                )
            raise LLMResponseError(
                f"LLM API çağrısında hata oluştu (HTTP {status_code or 'bilinmiyor'})."
            )
        except openai.APIError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM API genel hatası: %s. Model: %s, Süre: %.1fms",
                type(e).__name__,
                self.model,
                duration_ms,
            )
            raise LLMResponseError(f"LLM API hatası: {type(e).__name__} - {e}")
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM API beklenmeyen hata: %s. Model: %s, Süre: %.1fms",
                type(e).__name__,
                self.model,
                duration_ms,
            )
            raise LLMResponseError(f"LLM API çağrısında beklenmeyen hata: {type(e).__name__}")

        duration_ms = (time.perf_counter() - start_time) * 1000

        if not response.choices:
            raise LLMResponseError("LLM API yanıtında hiçbir seçenek (choice) dönmedi.")

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        if finish_reason == "length":
            raise LLMResponseError("Yanıt uzunluk sınırında kesildi")

        if finish_reason not in self.VALID_FINISH_REASONS:
            logger.warning("Beklenmeyen finish_reason alındı: %s", finish_reason)
            raise LLMResponseError(f"Beklenmeyen model durma nedeni: '{finish_reason}'")

        msg = choice.message
        text = getattr(msg, "content", "") or ""

        # Tool calls ayrıştırma
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        parsed_tool_calls: list[dict[str, Any]] = []
        sanitized_tool_calls: list[dict[str, Any]] = []

        for tc in raw_tool_calls:
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")
            fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
            fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
            fn_args_raw = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else "")

            if isinstance(fn_args_raw, str):
                try:
                    fn_args = json.loads(fn_args_raw)
                except Exception:
                    fn_args = {"_parse_error": True, "raw": fn_args_raw}
            elif isinstance(fn_args_raw, dict):
                fn_args = fn_args_raw
            else:
                fn_args = {}

            parsed_tool_calls.append({
                "id": tc_id,
                "name": fn_name,
                "arguments": fn_args,
            })

            # Sonraki tura aktarılacak OpenAI standardı tool_call dict'i
            call_dict = {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": fn_args_raw if isinstance(fn_args_raw, str) else json.dumps(fn_args, ensure_ascii=False),
                },
            }
            sanitized_tool_calls.append(call_dict)

        # Mesaj sanitizasyonu: Sadece role, content ve varsa tool_calls içeren temiz dict oluştur.
        # reasoning veya reasoning_content alanları kesinlikle dahil edilmez.
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if sanitized_tool_calls:
            assistant_message["tool_calls"] = sanitized_tool_calls

        # Usage bilgisi
        usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            usage_dict = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        logger.info(
            "LLM çağrısı tamamlandı. Model: %s, Süre: %.1fms, Girdi: %d token, Çıktı: %d token, Stop: %s",
            self.model,
            duration_ms,
            usage_dict["prompt_tokens"],
            usage_dict["completion_tokens"],
            finish_reason,
        )

        return LLMResponse(
            text=text,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            usage=usage_dict,
            message=assistant_message,
        )
