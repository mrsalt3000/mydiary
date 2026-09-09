/*
 * Local Diary — exports management / processing page behaviour (Task 16).
 *
 * Scope:
 *  - the weekly export form (GET-create through POST /api/v1/exports/weeks/…);
 *  - "diary action" forms (normalize / summarize / wellbeing / re-export):
 *    an open button reveals a confirmation panel whose submit stays disabled
 *    until every required checkbox (external consent, wellbeing disclaimer,
 *    revision warning) is checked; the submit POSTs to the JSON API;
 *  - job polling: a launched pipeline job is polled through
 *    GET /api/v1/jobs/{id} until terminal — success reloads the page (fresh
 *    lineage/statuses), failure shows the safe error summary;
 *  - resuming the polling of jobs that were in flight when the page loaded
 *    (data-active-job on the form);
 *  - the /processing "retry" buttons (POST /api/v1/jobs/{id}/retry).
 *
 * Safety rules (preserved from app.js, must hold for any later edit):
 *  - no HTML-string sinks anywhere: statuses, errors and links are built with
 *    textContent / createElement only; diary content is never inserted;
 *  - error strings shown to the user are static, or the server-provided safe
 *    error_summary rendered via textContent (never markup); raw provider
 *    payloads and stack traces never reach the page;
 *  - all requests go to this application only (no external hosts).
 */
"use strict";

(function () {
  var POLL_INTERVAL_MS = 2000;
  var POLL_LIMIT = 450; // ~15 minutes at 2 s, then ask the user to reload
  var GENERIC_ERROR = "Не удалось выполнить действие. Попробуйте ещё раз.";

  var JOB_STATUS_TEXT = {
    pending: "В очереди…",
    retrying: "Ожидает повтора…",
    running: "Выполняется…",
    succeeded: "Готово",
    failed: "Ошибка",
    cancelled: "Отменено"
  };

  var ACTION_ERRORS = {
    normalize: {
      "404": "Выгрузка не найдена.",
      "409": "Файл оригинала выгрузки отсутствует — запуск невозможен."
    },
    summarize: {
      "404": "Выгрузка не найдена.",
      "409": "Исходный файл выгрузки отсутствует — запуск невозможен."
    },
    wellbeing: {
      "403": "Wellbeing-анализ отключён в настройках приложения.",
      "404": "Сначала выполните нормализацию этой выгрузки.",
      "409": "Требуется явное подтверждение запуска.",
      "422": "Не удалось запустить анализ: запрос не прошёл проверку. " +
        "Обновите страницу и попробуйте ещё раз."
    },
    reexport: {
      "409": "Не удалось создать новую ревизию: неделя ещё не выгружена " +
        "или такая ревизия уже существует.",
      "422": "Не удалось создать ревизию: некорректная дата недели."
    }
  };

  function byRole(form, name) {
    return form.querySelector('[data-role="' + name + '"]');
  }

  function setText(element, text) {
    if (element) {
      element.textContent = text;
    }
  }

  function show(element) {
    if (element) {
      element.hidden = false;
    }
  }

  function hide(element) {
    if (element) {
      element.hidden = true;
    }
  }

  function clearFeedback(form) {
    var status = byRole(form, "status");
    var error = byRole(form, "error");
    setText(status, "");
    setText(error, "");
    hide(status);
    hide(error);
  }

  function showStatus(form, text) {
    var status = byRole(form, "status");
    setText(status, text);
    if (text) {
      show(status);
    } else {
      hide(status);
    }
  }

  function showError(form, text) {
    var error = byRole(form, "error");
    setText(error, text);
    show(error);
  }

  function allRequiredChecked(form) {
    var checks = form.querySelectorAll(
      'input[data-role="confirm-external"], ' +
      'input[data-role="confirm-wellbeing"], ' +
      'input[data-role="confirm-extra"]'
    );
    for (var index = 0; index < checks.length; index += 1) {
      if (!checks[index].checked) {
        return false;
      }
    }
    return true;
  }

  function setBusy(form, busy) {
    var open = byRole(form, "open");
    var submit = byRole(form, "submit");
    var label = form.getAttribute("data-label") || "Запустить";
    if (busy) {
      form.setAttribute("data-busy", "1");
    } else {
      form.removeAttribute("data-busy");
    }
    if (open && !form.getAttribute("data-active-job")) {
      open.disabled = busy;
    }
    if (submit) {
      submit.disabled = busy || !allRequiredChecked(form);
      submit.textContent = busy ? "Запуск…" : (submit.getAttribute("data-idle-label") || label);
    }
  }

  function syncSubmit(form) {
    var submit = byRole(form, "submit");
    if (submit) {
      submit.disabled = form.getAttribute("data-busy") === "1" || !allRequiredChecked(form);
    }
  }

  function openPanel(form) {
    clearFeedback(form);
    hide(byRole(form, "open"));
    show(byRole(form, "panel"));
    syncSubmit(form);
  }

  function cancelPanel(form) {
    clearFeedback(form);
    hide(byRole(form, "panel"));
    show(byRole(form, "open"));
    setBusy(form, false);
    form.removeAttribute("data-active-job");
    var submit = byRole(form, "submit");
    if (submit && submit.getAttribute("data-idle-label")) {
      submit.textContent = submit.getAttribute("data-idle-label");
    }
  }

  /* ------------------------------------------------------------------ *
   * Job polling (the single Task-16 status pattern).
   * ------------------------------------------------------------------ */

  function pollJob(form, jobId) {
    form.setAttribute("data-active-job", jobId);
    showStatus(form, "Отправка…");
    setBusy(form, true);

    var remaining = POLL_LIMIT;
    function tick() {
      remaining -= 1;
      if (remaining <= 0) {
        finishFailed(form, "Статус недоступен. Обновите страницу и проверьте «Обработку».");
        return;
      }
      fetch("/api/v1/jobs/" + encodeURIComponent(jobId))
        .then(function (response) {
          if (!response.ok) {
            throw new Error("job-poll-failed");
          }
          return response.json();
        })
        .then(function (job) {
          if (job.status === "pending" || job.status === "retrying") {
            showStatus(form, JOB_STATUS_TEXT.pending);
            window.setTimeout(tick, POLL_INTERVAL_MS);
          } else if (job.status === "running") {
            showStatus(form, JOB_STATUS_TEXT.running);
            window.setTimeout(tick, POLL_INTERVAL_MS);
          } else if (job.status === "succeeded") {
            finishSucceeded(form);
          } else {
            // failed / cancelled / unknown terminal state
            finishFailed(form, job.error_summary || "Задача завершилась ошибкой.");
          }
        })
        .catch(function () {
          window.setTimeout(tick, POLL_INTERVAL_MS); // transient network error
        });
    }
    tick();
  }

  function finishSucceeded(form) {
    showStatus(form, "Готово — обновляем страницу…");
    form.removeAttribute("data-active-job");
    window.setTimeout(function () {
      window.location.reload();
    }, 350);
  }

  function finishFailed(form, summary) {
    form.removeAttribute("data-active-job");
    setBusy(form, false);
    var message = summary || "Задача завершилась ошибкой. Подробности — на странице «Обработка».";
    if (summary) {
      message = "Задача завершилась ошибкой: " + summary;
    }
    showError(form, message);
  }

  function startAction(form) {
    if (!allRequiredChecked(form)) {
      showError(form, "Подтвердите все условия запуска.");
      return;
    }
    clearFeedback(form);
    var action = form.getAttribute("data-action") || "";
    if (action === "reexport") {
      startReexport(form);
    } else {
      startPipeline(form);
    }
  }

  function startPipeline(form) {
    var url = form.getAttribute("data-url") || "";
    var exportId = form.getAttribute("data-export-id") || "";
    var action = form.getAttribute("data-action") || "";
    var payload = { export_id: exportId };
    if (action === "wellbeing") {
      // Wellbeing runs require the server-side confirm flag as well. The flag
      // is taken from the acknowledgement checkbox, which must be checked
      // before the submit button enables (mirrors app.js on the wellbeing
      // page): unchecked → no confirm → the server rejects with 409.
      var wellbeingCheck = form.querySelector('input[data-role="confirm-wellbeing"]');
      payload.confirm = !!(wellbeingCheck && wellbeingCheck.checked);
    }
    setBusy(form, true);
    showStatus(form, "Отправка…");
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        if (response.status === 202 || response.status === 200) {
          return response.json();
        }
        var mapping = ACTION_ERRORS[action] || {};
        throw { userMessage: mapping[String(response.status)] || GENERIC_ERROR };
      })
      .then(function (job) {
        pollJob(form, job.id);
      })
      .catch(function (error) {
        var message = (error && error.userMessage) || GENERIC_ERROR;
        setBusy(form, false);
        showError(form, message);
      });
  }

  function startReexport(form) {
    var url = form.getAttribute("data-url") || "";
    var action = form.getAttribute("data-action") || "";
    setBusy(form, true);
    showStatus(form, "Отправка…");
    fetch(url, { method: "POST" })
      .then(function (response) {
        if (response.status === 201) {
          return response.json();
        }
        var mapping = ACTION_ERRORS[action] || {};
        throw { userMessage: mapping[String(response.status)] || GENERIC_ERROR };
      })
      .then(function (created) {
        if (created && created.id) {
          showStatus(form, "Ревизия создана — открываем…");
          window.setTimeout(function () {
            window.location.href = "/exports/" + encodeURIComponent(created.id);
          }, 350);
        }
      })
      .catch(function (error) {
        var message = (error && error.userMessage) || GENERIC_ERROR;
        setBusy(form, false);
        showError(form, message);
      });
  }

  /* ------------------------------------------------------------------ *
   * Wiring the diary-action forms.
   * ------------------------------------------------------------------ */

  function wireActionForm(form) {
    var open = byRole(form, "open");
    var cancel = byRole(form, "cancel");
    var submit = byRole(form, "submit");
    if (!open || !cancel || !submit) {
      return;
    }
    submit.setAttribute("data-idle-label", submit.textContent);
    open.addEventListener("click", function () {
      openPanel(form);
    });
    cancel.addEventListener("click", function () {
      cancelPanel(form);
    });
    form.addEventListener("change", function (event) {
      var target = event.target;
      if (target && target.getAttribute && target.getAttribute("data-role")) {
        syncSubmit(form);
      }
    });
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      startAction(form);
    });
  }

  function resumeActiveActions() {
    var forms = document.querySelectorAll("form.diary-action[data-active-job]");
    for (var index = 0; index < forms.length; index += 1) {
      var form = forms[index];
      hide(byRole(form, "open"));
      show(byRole(form, "panel"));
      pollJob(form, form.getAttribute("data-active-job"));
    }
  }

  function wireDiaryActions() {
    var forms = document.querySelectorAll("form.diary-action");
    for (var index = 0; index < forms.length; index += 1) {
      wireActionForm(forms[index]);
    }
    if (document.readyState === "loading") {
      // forms are already in the DOM (server-rendered); waiting only defers
      // the resume until listeners exist — handled below via DOMContentLoaded.
      document.addEventListener("DOMContentLoaded", resumeActiveActions);
    } else {
      resumeActiveActions();
    }
  }

  /* ------------------------------------------------------------------ *
   * Weekly export form.
   * ------------------------------------------------------------------ */

  function wireExportWeekForms() {
    var forms = document.querySelectorAll("form.export-week-form");
    for (var index = 0; index < forms.length; index += 1) {
      wireExportWeekForm(forms[index]);
    }
  }

  function wireExportWeekForm(form) {
    var input = form.querySelector('input[name="sunday"]');
    var error = form.querySelector('[data-role="error"]');
    var button = form.querySelector('button[type="submit"]');
    if (!input || !error || !button) {
      return;
    }
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var sunday = input.value;
      if (!sunday) {
        setText(error, "Укажите дату воскресенья недели, которую нужно выгрузить.");
        show(error);
        return;
      }
      setText(error, "");
      hide(error);
      button.disabled = true;
      fetch("/api/v1/exports/weeks/" + encodeURIComponent(sunday), { method: "POST" })
        .then(function (response) {
          if (response.status === 201) {
            window.setTimeout(function () {
              window.location.reload();
            }, 300);
            return null;
          }
          if (response.status === 409) {
            throw { userMessage: "Не удалось выгрузить: в этой неделе нет записей, " +
              "либо неделя уже выгружена. Для новой ревизии используйте «Новая ревизия»." };
          }
          if (response.status === 422) {
            throw { userMessage: "Указанная дата не является датой воскресенья недели." };
          }
          throw { userMessage: "Не удалось выгрузить неделю. Попробуйте ещё раз." };
        })
        .catch(function (err) {
          var message = (err && err.userMessage) || "Не удалось выгрузить неделю.";
          setText(error, message);
          show(error);
        })
        .finally(function () {
          button.disabled = false;
        });
    });
  }

  /* ------------------------------------------------------------------ *
   * Processing-page job retry (delegated: the rows are htmx-polled).
   * ------------------------------------------------------------------ */

  function handleRetrySubmit(event) {
    var form = event.target;
    if (!form || typeof form.closest !== "function") {
      return;
    }
    form = form.closest("form.job-retry");
    if (!form) {
      return;
    }
    event.preventDefault();
    var url = form.getAttribute("data-url") || "";
    var button = form.querySelector('button[type="submit"]');
    var error = form.querySelector('[data-role="error"]');
    if (!url || !error) {
      return;
    }
    setText(error, "");
    hide(error);
    if (button) {
      button.disabled = true;
    }
    fetch(url, { method: "POST" })
      .then(function (response) {
        if (response.status === 202 || response.status === 200) {
          window.setTimeout(function () {
            window.location.reload();
          }, 300);
          return;
        }
        if (response.status === 409) {
          throw { userMessage: "Задача не может быть повторена: она ещё выполняется " +
            "или уже завершена успешно." };
        }
        throw { userMessage: "Не удалось повторить задачу. Попробуйте ещё раз." };
      })
      .catch(function (err) {
        var message = (err && err.userMessage) || "Не удалось повторить задачу.";
        setText(error, message);
        show(error);
        if (button) {
          button.disabled = false;
        }
      });
  }

  function onReady() {
    wireDiaryActions();
    wireExportWeekForms();
    document.addEventListener("submit", handleRetrySubmit);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onReady);
  } else {
    onReady();
  }
})();
