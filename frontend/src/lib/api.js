const API_BASE_URL = 'http://localhost:8000'

function toQueryString(params) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
        if (value === undefined || value === null || value === '') {
            return
        }
        searchParams.set(key, String(value))
    })
    return searchParams.toString()
}

async function request(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {}),
        },
        ...options,
    })

    const contentType = response.headers.get('content-type') || ''
    const payload = contentType.includes('application/json')
        ? await response.json()
        : await response.text()

    if (!response.ok) {
        const message =
            typeof payload === 'object' && payload !== null
                ? payload.detail || payload.message || JSON.stringify(payload)
                : payload
        throw new Error(message || `Request failed with status ${response.status}`)
    }

    return payload
}

function postJson(path, payload) {
    return request(path, {
        method: 'POST',
        body: JSON.stringify(payload),
    })
}

export function buildAbsoluteUrl(pathOrUrl) {
    if (!pathOrUrl || typeof pathOrUrl !== 'string') {
        return ''
    }
    if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
        return pathOrUrl
    }
    const normalized = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`
    return `${API_BASE_URL}${normalized}`
}

export function fetchHealth() {
    return request('/health')
}

export function fetchModels() {
    return request('/api/v1/models')
}

export function fetchModelDetails({ model_path } = {}) {
    const query = toQueryString({ model_path })
    return request(`/api/v1/models/details${query ? `?${query}` : ''}`)
}

export function runTrain(payload) {
    return postJson('/api/v1/train', payload)
}

export function runPredict(payload) {
    const hasCustomModel = Boolean(payload?.model_path)
    const endpoint = hasCustomModel ? '/api/v1/models/predict' : '/api/v1/predict'
    return postJson(endpoint, payload)
}

export function generateSyntheticData(payload) {
    return postJson('/api/v1/synthetic-data', payload)
}

export function fetchLatestReport() {
    return request('/api/v1/reports/latest')
}

export function fetchDataset(params) {
    const query = toQueryString(params)
    const suffix = query ? `?${query}` : ''
    return request(`/api/v1/dataset${suffix}`)
}
