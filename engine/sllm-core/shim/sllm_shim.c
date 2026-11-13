#include "sllm_shim.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

// ---- log capture ---------------------------------------------------------------------------
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static char *          g_log_buf   = NULL;
static size_t          g_log_len   = 0;
static size_t          g_log_cap   = 0;
static bool            g_log_on    = false;
static bool            g_log_echo  = false;

static void sllm_log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text == NULL) {
        return;
    }
    const size_t n = strlen(text);
    pthread_mutex_lock(&g_log_mutex);
    if (g_log_len + n + 1 > g_log_cap) {
        size_t cap = g_log_cap ? g_log_cap : 4096;
        while (cap < g_log_len + n + 1) {
            cap *= 2;
        }
        char * p = (char *) realloc(g_log_buf, cap);
        if (p == NULL) {
            pthread_mutex_unlock(&g_log_mutex);
            return;  // drop the message rather than abort
        }
        g_log_buf = p;
        g_log_cap = cap;
    }
    memcpy(g_log_buf + g_log_len, text, n);
    g_log_len += n;
    g_log_buf[g_log_len] = '\0';
    const bool echo = g_log_echo;
    pthread_mutex_unlock(&g_log_mutex);
    if (echo) {
        fputs(text, stderr);
    }
}

bool sllm_log_capture_begin(bool echo) {
    pthread_mutex_lock(&g_log_mutex);
    if (g_log_on) {
        pthread_mutex_unlock(&g_log_mutex);
        return false;
    }
    g_log_on   = true;
    g_log_echo = echo;
    g_log_len  = 0;
    if (g_log_buf) {
        g_log_buf[0] = '\0';
    }
    pthread_mutex_unlock(&g_log_mutex);
    ggml_log_set(sllm_log_cb, NULL);
    return true;
}

size_t sllm_log_capture_end(void) {
    ggml_log_set(NULL, NULL);  // NULL restores the default (stderr) logger
    pthread_mutex_lock(&g_log_mutex);
    g_log_on = false;
    const size_t n = g_log_len;
    pthread_mutex_unlock(&g_log_mutex);
    return n;
}

size_t sllm_log_capture_copy(char * dst, size_t cap) {
    pthread_mutex_lock(&g_log_mutex);
    const size_t n = g_log_len;
    if (dst != NULL && cap > 0) {
        const size_t m = n < cap - 1 ? n : cap - 1;
        if (m > 0) {
            memcpy(dst, g_log_buf, m);
        }
        dst[m] = '\0';
    }
    pthread_mutex_unlock(&g_log_mutex);
    return n;
}

// ---- device query --------------------------------------------------------------------------
bool sllm_dev_query(size_t i, struct sllm_dev_info * info) {
    if (i >= ggml_backend_dev_count()) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    struct ggml_backend_dev_props p;
    memset(&p, 0, sizeof(p));
    ggml_backend_dev_get_props(dev, &p);
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "%s", p.name ? p.name : "");
    snprintf(info->description, sizeof(info->description), "%s", p.description ? p.description : "");
    info->type                 = (int) p.type;
    info->memory_free          = p.memory_free;
    info->memory_total         = p.memory_total;
    info->async                = p.caps.async;
    info->host_buffer          = p.caps.host_buffer;
    info->buffer_from_host_ptr = p.caps.buffer_from_host_ptr;
    info->events               = p.caps.events;
    info->mmap_support         = p.caps.mmap_support;
    return true;
}

// ---- layout probes -------------------------------------------------------------------------
void sllm_layout_probe(struct sllm_layout * out) {
    out->sizeof_ggml_tensor     = sizeof(struct ggml_tensor);
    out->offsetof_type          = offsetof(struct ggml_tensor, type);
    out->offsetof_buffer        = offsetof(struct ggml_tensor, buffer);
    out->offsetof_ne            = offsetof(struct ggml_tensor, ne);
    out->offsetof_nb            = offsetof(struct ggml_tensor, nb);
    out->offsetof_op            = offsetof(struct ggml_tensor, op);
    out->offsetof_src           = offsetof(struct ggml_tensor, src);
    out->offsetof_data          = offsetof(struct ggml_tensor, data);
    out->offsetof_name          = offsetof(struct ggml_tensor, name);
    out->sizeof_dev_props       = sizeof(struct ggml_backend_dev_props);
    out->sizeof_type_traits     = sizeof(struct ggml_type_traits);
    out->offsetof_to_float      = offsetof(struct ggml_type_traits, to_float);
    out->sizeof_gguf_init_params = sizeof(struct gguf_init_params);
    out->ggml_max_dims          = GGML_MAX_DIMS;
    out->ggml_max_name          = GGML_MAX_NAME;
    out->ggml_max_src           = GGML_MAX_SRC;
}
