/*
 * Local Diary — chat page behaviour (Tasks 06/07).
 *
 * Scope: the text compose form (Task 06) and voice messages (Task 07):
 * recording through MediaRecorder when available, a file-upload fallback
 * through a hidden <input type=file>, upload to POST /api/v1/audio and
 * insertion of the created voice message into the chat list. The "load older
 * messages" paging stays declarative (htmx server-rendered fragments).
 *
 * Safety rules (must be preserved by any later edit):
 *  - Diary text and audio metadata are NEVER placed into the DOM as markup.
 *    New messages are built with createElement + textContent/properties only
 *    — no HTML-string sinks anywhere in this file.
 *  - The audio element <src> always points at the server-generated playback
 *    URL "/api/v1/audio/<uuid>/file" derived from the 201 response; no other
 *    value is ever accepted.
 *  - Error/info strings shown to the user are static; server-provided error
 *    payloads are never echoed into the page.
 *  - No fetch/script/link to external hosts: the page only talks to this app.
 */
"use strict";

(function () {
  var form = document.getElementById("compose-form");
  var textarea = document.getElementById("message-text");
  var errorBox = document.getElementById("form-error");
  var submitButton = document.getElementById("compose-submit");
  var messageList = document.getElementById("message-list");
  var emptyState = document.getElementById("message-empty");

  if (!form || !textarea || !errorBox || !submitButton || !messageList) {
    return;
  }

  var maxLength = Number(form.dataset.maxLength || 0);

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  /* Server timestamps are ISO-8601 with the application offset already
     applied (e.g. "2026-09-10T05:45:00+03:00"). Render the wall-clock part
     in the same "%d.%m.%Y %H:%M" shape the server uses — the offset is read
     from the string itself, so the browser's own timezone never matters. */
  function formatTimestamp(iso) {
    var match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
    if (!match) {
      return iso;
    }
    return match[3] + "." + match[2] + "." + match[1] + " " + match[4] + ":" + match[5];
  }

  function messageFooter(message) {
    var meta = document.createElement("header");
    meta.className = "message__meta";
    var time = document.createElement("time");
    time.dateTime = message.created_at;
    time.textContent = formatTimestamp(message.created_at);
    meta.appendChild(time);
    return meta;
  }

  function appendMessage(message) {
    if (emptyState && emptyState.parentNode === messageList) {
      emptyState.remove();
    }

    var article = document.createElement("article");
    article.className = "message";
    article.id = "message-" + message.id;

    var meta = messageFooter(message);
    var body = document.createElement("div");
    body.className = "message__body";
    body.textContent = message.body; // diary text: textContent only, never markup

    article.appendChild(meta);
    article.appendChild(body);
    messageList.appendChild(article);
    article.scrollIntoView({ block: "nearest" });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var body = textarea.value;

    if (!body.trim()) {
      showError("Запись не может быть пустой.");
      textarea.focus();
      return;
    }
    if (maxLength > 0 && body.length > maxLength) {
      showError("Запись слишком длинная — не более " + maxLength + " символов.");
      textarea.focus();
      return;
    }

    clearError();
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");

    fetch("/api/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: body }),
    })
      .then(function (response) {
        if (response.status === 201) {
          return response.json().then(function (created) {
            textarea.value = "";
            appendMessage(created);
          });
        }
        throw new Error("create-failed");
      })
      .catch(function () {
        // Static, friendly message. Server error details are never rendered.
        showError("Не удалось сохранить запись. Попробуйте ещё раз.");
      })
      .finally(function () {
        submitButton.disabled = false;
        form.removeAttribute("aria-busy");
        textarea.focus();
      });
  });

  /* ------------------------------------------------------------------ *
   * Voice messages (Task 07).
   * ------------------------------------------------------------------ */

  var voiceBox = document.getElementById("voice-compose");
  var recordButton = document.getElementById("voice-record");
  var stopButton = document.getElementById("voice-stop");
  var fileInput = document.getElementById("voice-file");
  var voiceStatus = document.getElementById("voice-status");
  var voiceError = document.getElementById("voice-error");
  var recordingLabel = document.getElementById("voice-recording");

  if (!voiceBox || !recordButton || !stopButton || !fileInput ||
      !voiceStatus || !voiceError || !recordingLabel) {
    return;
  }

  var maxAudioMb = Number(voiceBox.dataset.maxAudioMb || 0);
  var uploadInFlight = false;
  var recorder = null;
  var liveStream = null;
  var chunks = [];
  var preferredMimeTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];

  function showVoiceStatus(message) {
    voiceStatus.textContent = message;
    voiceStatus.hidden = false;
  }

  function hideVoiceStatus() {
    voiceStatus.textContent = "";
    voiceStatus.hidden = true;
  }

  function showVoiceError(message) {
    voiceError.textContent = message;
    voiceError.hidden = false;
  }

  function hideVoiceError() {
    voiceError.textContent = "";
    voiceError.hidden = true;
  }

  function setVoiceControlsDisabled(disabled) {
    recordButton.disabled = disabled;
    stopButton.disabled = disabled;
    fileInput.disabled = disabled;
  }

  function recorderSupported() {
    return Boolean(
      window.MediaRecorder &&
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia
    );
  }

  function stopTracks(activeStream) {
    if (!activeStream) {
      return;
    }
    activeStream.getTracks().forEach(function (track) {
      track.stop();
    });
  }

  function uploadAudio(blob, fileName) {
    if (uploadInFlight) {
      return;
    }
    uploadInFlight = true;
    hideVoiceError();
    showVoiceStatus("Отправка…");
    setVoiceControlsDisabled(true);

    var formData = new FormData();
    formData.append("file", blob, fileName);

    fetch("/api/v1/audio", {
      method: "POST",
      body: formData, // no Content-Type header: the boundary is set by fetch
    })
      .then(function (response) {
        if (response.status === 201) {
          return response.json().then(function (created) {
            hideVoiceStatus();
            appendAudioMessage(created);
          });
        }
        // Static messages keyed on the status; the server payload is never
        // rendered into the page.
        if (response.status === 413) {
          throw { userMessage: "Файл слишком большой — не более " + maxAudioMb + " МБ." };
        }
        if (response.status === 400 || response.status === 422) {
          throw { userMessage: "Не удалось загрузить: неподдерживаемый формат или повреждённый файл." };
        }
        throw { userMessage: "Не удалось загрузить запись. Попробуйте ещё раз." };
      })
      .catch(function (error) {
        var message = (error && error.userMessage) ||
          "Не удалось загрузить запись. Попробуйте ещё раз.";
        showVoiceError(message);
      })
      .finally(function () {
        uploadInFlight = false;
        setVoiceControlsDisabled(false);
      });
  }

  function appendAudioMessage(payload) {
    if (emptyState && emptyState.parentNode === messageList) {
      emptyState.remove();
    }

    var article = document.createElement("article");
    article.className = "message message--audio";
    article.id = "message-" + payload.message_id;

    var meta = messageFooter(payload);
    var badge = document.createElement("span");
    badge.className = "message__badge";
    badge.textContent = "Голосовое";
    meta.appendChild(badge);

    var box = document.createElement("div");
    box.className = "message__audio";

    var player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    // Server-generated playback URL derived from the response id only.
    player.src = "/api/v1/audio/" + payload.id + "/file";
    box.appendChild(player);

    var caption = document.createElement("p");
    caption.className = "message__caption";
    caption.textContent = "Не расшифровано"; // Task 07: manual transcription is Task 09.
    box.appendChild(caption);

    article.appendChild(meta);
    article.appendChild(box);
    messageList.appendChild(article);
    article.scrollIntoView({ block: "nearest" });
  }

  function pickMimeType() {
    for (var index = 0; index < preferredMimeTypes.length; index += 1) {
      if (window.MediaRecorder.isTypeSupported(preferredMimeTypes[index])) {
        return preferredMimeTypes[index];
      }
    }
    return "";
  }

  if (!recorderSupported()) {
    // Upload-only fallback: keep the file control, drop the record button.
    recordButton.hidden = true;
  }
  voiceBox.hidden = false; // JS (and at least one input path) is available.

  recordButton.addEventListener("click", function () {
    if (recorder) {
      return; // already recording
    }
    hideVoiceError();
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(function (media) {
        liveStream = media;
        chunks = [];
        var mimeType = pickMimeType();
        var options = {};
        if (mimeType) {
          options.mimeType = mimeType;
        }
        try {
          recorder = new MediaRecorder(media, options);
        } catch (exception) {
          recorder = null;
          stopTracks(liveStream);
          liveStream = null;
          showVoiceError("Запись недоступна в этом браузере — загрузите файл.");
          return;
        }
        recorder.addEventListener("dataavailable", function (event) {
          if (event.data && event.data.size > 0) {
            chunks.push(event.data);
          }
        });
        recorder.addEventListener("stop", function () {
          var finishedType = recorder.mimeType || mimeType || "audio/webm";
          var finishedStream = liveStream;
          recorder = null;
          liveStream = null;
          stopTracks(finishedStream);
          recordButton.hidden = false;
          stopButton.hidden = true;
          recordingLabel.hidden = true;
          if (!chunks.length) {
            showVoiceError("Запись пуста. Попробуйте ещё раз.");
            chunks = [];
            return;
          }
          var blob = new Blob(chunks, { type: finishedType });
          chunks = [];
          var fileName = finishedType.indexOf("ogg") !== -1
            ? "recording.ogg"
            : "recording.webm";
          uploadAudio(blob, fileName);
        });
        recorder.start();
        recordButton.hidden = true;
        stopButton.hidden = false;
        recordingLabel.hidden = false;
      })
      .catch(function () {
        showVoiceError("Микрофон недоступен или доступ запрещён. Можно загрузить файл.");
      });
  });

  stopButton.addEventListener("click", function () {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  });

  fileInput.addEventListener("change", function () {
    var file = fileInput.files && fileInput.files[0];
    if (!file) {
      return;
    }
    // The user-chosen file name goes only into the multipart part label; the
    // server never uses it (storage is under a server-generated UUID).
    uploadAudio(file, file.name);
    fileInput.value = "";
  });
})();
