import { useEffect, useMemo, useState } from 'react'
import { Info, Hash, Shuffle } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  buildAbsoluteUrl,
  fetchDataset,
  fetchHealth,
  fetchLatestReport,
  fetchModelDetails,
  fetchModels,
  generateSyntheticData,
  runPredict,
  runTrain,
} from '@/lib/api'

const TRAIN_DEFAULTS = {
  test_size: '0.2',
  random_state: '42',
}

const METRIC_KEYS = ['mae', 'mse', 'rmse', 'r2', 'mape']
const THEME_STORAGE_KEY = 'dashboard-theme'
const NO_MODELS_VALUE = '__no_models__'
const NO_MODELS_MESSAGE = 'No models trained yet. Train one to start prediction.'
const DATASET_SOURCE_URL = 'https://www.kaggle.com/datasets/yasserh/bigmartsalesdataset'
const DATASET_COLUMN_INFO = [
  {
    column: 'Item_Identifier',
    description: 'Unique Number assigned to each Item',
  },
  {
    column: 'Item_Weight',
    description: 'Item Weight in g',
  },
  {
    column: 'Item_Fat_Content',
    description: 'Item Fat Content',
  },
  {
    column: 'Item_Visibility',
    description: 'Placement value of each item: 0 - Far & Behind 1 - Near & Front',
  },
  {
    column: 'Item_Type',
    description: 'Type of item utility',
  },
  {
    column: 'Item_MRP',
    description: 'Price of the Item',
  },
  {
    column: 'Outlet_Identifier',
    description: 'Unique Outler Name',
  },
  {
    column: 'Outlet_Establishment_Year',
    description: 'Year of Outlet Establishment',
  },
  {
    column: 'Outlet_Size',
    description: 'Size of the Outler',
  },
  {
    column: 'Outlet_Location_Type',
    description: 'Tier of Outler Location',
  },
]
const DATASET_COLUMN_INFO_MAP = Object.fromEntries(
  DATASET_COLUMN_INFO.map((item) => [item.column, item.description]),
)

function numberOrUndefined(value) {
  if (value === '' || value === null || value === undefined) {
    return undefined
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function prettyMetricName(key) {
  return key.toUpperCase()
}

function getDatasetColumnDescription(column) {
  return DATASET_COLUMN_INFO_MAP[column] ?? ''
}

function getTrainingMetrics(result) {
  return result?.stats ?? result?.performance?.metrics ?? {}
}

function toGraphList(result) {
  const list = []

  const perfGraphs = Array.isArray(result?.performance?.graphs)
    ? result.performance.graphs
    : []

  perfGraphs.forEach((graph, index) => {
    if (!graph?.url) {
      return
    }
    list.push({
      id: graph.id ?? `performance-${index}`,
      title: graph.file_name ?? graph.id ?? `Graph ${index + 1}`,
      url: buildAbsoluteUrl(graph.url),
      source: 'performance.graphs',
    })
  })

  const appendFromMap = (sourceKey, map) => {
    if (!map || typeof map !== 'object') {
      return
    }
    Object.entries(map).forEach(([key, value]) => {
      if (!value || typeof value !== 'string') {
        return
      }
      list.push({
        id: `${sourceKey}-${key}`,
        title: key,
        url: buildAbsoluteUrl(value),
        source: sourceKey,
      })
    })
  }

  appendFromMap('temp_plot_urls', result?.temp_plot_urls)
  appendFromMap('plot_urls', result?.plot_urls)

  const seen = new Set()
  return list.filter((graph) => {
    if (!graph.url || seen.has(graph.url)) {
      return false
    }
    seen.add(graph.url)
    return true
  })
}

function formatPathForDisplay(path) {
  if (!path || typeof path !== 'string') {
    return '--'
  }

  const normalized = path.replace(/\\/g, '/').trim()
  const isAbsolute = normalized.startsWith('/')
  const parts = normalized.split('/').filter(Boolean)

  if (parts.length <= 3 && normalized.length <= 48) {
    return normalized
  }

  if (parts.length >= 4) {
    const first = parts[0]
    const tail = parts.slice(-2).join('/')
    return `${isAbsolute ? '/' : ''}${first}/.../${tail}`
  }

  return `${normalized.slice(0, 14)}...${normalized.slice(-20)}`
}

function PathValue({ value }) {
  const displayValue = formatPathForDisplay(value)

  if (!value || typeof value !== 'string') {
    return <span>{displayValue}</span>
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-block max-w-full cursor-help rounded bg-muted/30 px-1.5 py-0.5 font-mono text-xs text-foreground/90">
          {displayValue}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-2xl break-all font-mono text-xs">
        {value}
      </TooltipContent>
    </Tooltip>
  )
}

function normalizeModelEntry(model) {
  if (!model || typeof model !== 'object') {
    return null
  }

  const sizeBytes = Number(model.size_bytes)

  return {
    ...model,
    model_path: model.model_path ?? '',
    model_name: model.model_name ?? '',
    model_stem: model.model_stem ?? '',
    size_bytes: Number.isFinite(sizeBytes) ? sizeBytes : null,
    created_at: model.created_at ?? '',
    modified_at: model.modified_at ?? '',
    report_dir: model.report_dir ?? '',
    metrics_path: model.metrics_path ?? '',
    has_report: Boolean(model.has_report),
    dataset_path: model.dataset_path ?? '',
    test_size: model.test_size ?? null,
    random_state: model.random_state ?? null,
    metrics: model.metrics && typeof model.metrics === 'object' ? model.metrics : {},
    plots: model.plots ?? [],
    feature_names: Array.isArray(model.feature_names) ? model.feature_names : [],
    target: model.target ?? '',
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '--'
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unitIndex = 0

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function getModelDisplayName(model) {
  return model?.model_name || model?.model_stem || model?.model_path || 'Model'
}

function isModelNotFoundMessage(message) {
  if (!message || typeof message !== 'string') {
    return false
  }

  return /model\s+file\s+not\s+found/i.test(message)
}

function normalizeModelUiMessage(message) {
  return isModelNotFoundMessage(message) ? NO_MODELS_MESSAGE : message
}

function safeJsonStringify(value) {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return ''
  }
}

function formatDisplayValue(value) {
  if (value === null || value === undefined || value === '') {
    return '--'
  }

  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  if (Array.isArray(value)) {
    if (value.every((item) => typeof item === 'string' || typeof item === 'number')) {
      return value.join(', ')
    }

    return `${value.length} item${value.length === 1 ? '' : 's'}`
  }

  if (typeof value === 'object') {
    return safeJsonStringify(value) || '--'
  }

  return String(value)
}

function GraphCard({ title, source, url, onOpen, broken, onBroken }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <CardDescription className="text-xs">{source}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {broken ? (
          <Alert variant="destructive">
            <AlertDescription>Image failed to load: {url}</AlertDescription>
          </Alert>
        ) : (
          <img
            src={url}
            alt={title}
            className="h-44 w-full rounded-md border object-cover"
            loading="lazy"
            onError={() => onBroken(url)}
          />
        )}
        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={() => onOpen({ title, source, url })}>
            View larger
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function SummaryCard({ label, value, description }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1">
        <p className="text-lg font-semibold">{formatDisplayValue(value)}</p>
        {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
      </CardContent>
    </Card>
  )
}

function App() {
  const [theme, setTheme] = useState(localStorage.getItem(THEME_STORAGE_KEY) || 'dark')
  const [health, setHealth] = useState({ status: 'loading', error: null })
  const [trainForm, setTrainForm] = useState(TRAIN_DEFAULTS)
  const [trainResult, setTrainResult] = useState(null)
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainError, setTrainError] = useState('')
  const [trainMetrics, setTrainMetrics] = useState(null)
  const [trainGraphs, setTrainGraphs] = useState([])
  
  const [datasetPage, setDatasetPage] = useState(1)
  const [datasetPageSize, setDatasetPageSize] = useState('20')
  const [datasetResult, setDatasetResult] = useState(null)
  const [datasetLoading, setDatasetLoading] = useState(false)
  const [datasetError, setDatasetError] = useState('')
  
  const [selectedModelPath, setSelectedModelPath] = useState('')
  const [selectedModelDetails, setSelectedModelDetails] = useState(null)
  const [selectedModelLoading, setSelectedModelLoading] = useState(false)
  const [selectedModelError, setSelectedModelError] = useState('')
  
  const [modelsCatalog, setModelsCatalog] = useState({ models: [], model_count: 0 })
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState('')
  
  const [predictInput, setPredictInput] = useState('')
  const [predictResult, setPredictResult] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState('')
  
  const [syntheticForm, setSyntheticForm] = useState({ count: '', random_state: '', include_target: false })
  const [syntheticResult, setSyntheticResult] = useState(null)
  const [syntheticLoading, setSyntheticLoading] = useState(false)
  const [syntheticError, setSyntheticError] = useState('')
  
  const [reportResult, setReportResult] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [reportMetrics, setReportMetrics] = useState(null)
  
  const [brokenImages, setBrokenImages] = useState({})
  const [viewerGraph, setViewerGraph] = useState(null)
  const [datasetInfoOpen, setDatasetInfoOpen] = useState(false)

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('/api/v1/health')
        if (!response.ok) throw new Error(`Health check failed: ${response.status}`)
        setHealth({ status: 'ok', error: null })
      } catch (error) {
        setHealth({ status: 'error', error: error.message })
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const loadModels = async () => {
      setModelsLoading(true)
      setModelsError('')
      try {
        const response = await fetchModels()
        const normalized = (response.models ?? []).map(normalizeModelEntry).filter(Boolean)
        setModelsCatalog({ models: normalized, model_count: normalized.length })
      } catch (error) {
        setModelsError(error.message)
        setModelsCatalog({ models: [], model_count: 0 })
      } finally {
        setModelsLoading(false)
      }
    }
    loadModels()
  }, [])

  useEffect(() => {
    if (!selectedModelPath || selectedModelPath === NO_MODELS_VALUE) {
      setSelectedModelDetails(null)
      setSelectedModelError('')
      return
    }
    const loadDetails = async () => {
      setSelectedModelLoading(true)
      setSelectedModelError('')
      try {
        const response = await fetchModelDetails({ model_path: selectedModelPath })
        setSelectedModelDetails(response)
      } catch (error) {
        setSelectedModelError(error.message)
        setSelectedModelDetails(null)
      } finally {
        setSelectedModelLoading(false)
      }
    }
    loadDetails()
  }, [selectedModelPath])

  useEffect(() => {
    const metrics = getTrainingMetrics(trainResult)
    setTrainMetrics(metrics)
  }, [trainResult])

  useEffect(() => {
    const graphs = toGraphList(trainResult)
    setTrainGraphs(graphs)
  }, [trainResult])

  useEffect(() => {
    const plots = toGraphList(reportResult)
    const reportMetricsData = getTrainingMetrics(reportResult)
    setReportMetrics(reportMetricsData)
  }, [reportResult])

  const hasAnyModel = modelsCatalog.models.length > 0
  const activeModel = selectedModelPath && selectedModelPath !== NO_MODELS_VALUE
    ? modelsCatalog.models.find((m) => m.model_path === selectedModelPath)
    : null
  const activeModelPath = activeModel?.model_path
  const reportPlots =
    reportResult?.plot_urls && typeof reportResult.plot_urls === 'object'
      ? Object.entries(reportResult.plot_urls)
      : []

  function updateTrainField(field, value) {
    setTrainForm((previous) => ({ ...previous, [field]: value }))
  }

  function updateSyntheticField(field, value) {
    setSyntheticForm((previous) => ({ ...previous, [field]: value }))
  }

  async function handleTrain(event) {
    event.preventDefault()
    setTrainLoading(true)
    setTrainError('')
    setBrokenImages({})

    const payload = {
      test_size: numberOrUndefined(trainForm.test_size),
      random_state: numberOrUndefined(trainForm.random_state),
    }

    try {
      const response = await runTrain(payload)
      setTrainResult(response)
    } catch (error) {
      setTrainError(error.message)
    } finally {
      setTrainLoading(false)
    }
  }

  async function handleFetchDataset() {
    setDatasetLoading(true)
    setDatasetError('')

    try {
      const response = await fetchDataset({
        page: datasetPage,
        page_size: Number(datasetPageSize),
      })
      setDatasetResult(response)
    } catch (error) {
      setDatasetError(error.message)
    } finally {
      setDatasetLoading(false)
    }
  }

  async function handlePredict() {
    setPredictLoading(true)
    setPredictError('')

    try {
      const parsed = JSON.parse(predictInput)
      const isObject = parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      const isArray = Array.isArray(parsed)

      if (!isObject && !isArray) {
        throw new Error('Records must be a JSON object or an array of objects.')
      }

      const customModelPath = selectedModelPath || ''
      const response = await runPredict(
        customModelPath ? { records: parsed, model_path: customModelPath } : { records: parsed },
      )
      setPredictResult(response)
    } catch (error) {
      setPredictError(normalizeModelUiMessage(error.message))
    } finally {
      setPredictLoading(false)
    }
  }

  async function handleGenerateSyntheticData() {
    setSyntheticLoading(true)
    setSyntheticError('')

    try {
      const response = await generateSyntheticData({
        count: numberOrUndefined(syntheticForm.count),
        random_state: numberOrUndefined(syntheticForm.random_state),
        include_target: syntheticForm.include_target,
      })

      setSyntheticResult(response)

      const records = response?.records
      const serializedRecords = safeJsonStringify(records)

      if (serializedRecords) {
        setPredictInput(serializedRecords)
      }
    } catch (error) {
      setSyntheticError(error.message)
    } finally {
      setSyntheticLoading(false)
    }
  }

  async function handleFetchLatestReport() {
    setReportLoading(true)
    setReportError('')

    try {
      const response = await fetchLatestReport()
      setReportResult(response)
    } catch (error) {
      setReportError(error.message)
    } finally {
      setReportLoading(false)
    }
  }

  function markImageBroken(url) {
    setBrokenImages((previous) => ({ ...previous, [url]: true }))
  }

  function openGraphViewer(graph) {
    setViewerGraph(graph)
  }

  const datasetColumns = datasetResult?.columns ?? []
  const datasetRows = datasetResult?.data ?? []
  const datasetPagination = datasetResult?.pagination

  function toggleTheme() {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
  }

  return (
    <main className="min-h-screen bg-linear-to-b from-muted/20 via-background to-background px-4 py-8 md:px-8">
      <div className="mx-auto w-full max-w-7xl space-y-6">
        <section className="flex flex-col gap-3 rounded-xl border bg-card/80 p-4 shadow-sm md:flex-row md:items-center md:justify-between md:p-6">
          <div>
            <h1 className="font-heading text-2xl font-semibold tracking-tight md:text-3xl">
              Big Mart Sales Dashboard
            </h1>
            <p className="text-sm text-muted-foreground">
              Train, inspect models, browse data, and run predictions from one place.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={toggleTheme}>
              {theme === 'dark' ? 'Switch to Light' : 'Switch to Dark'}
            </Button>
            <Badge variant={health.status === 'ok' ? 'default' : 'destructive'}>
              Backend: {health.status}
            </Badge>
            {health.error ? (
              <span className="max-w-xs text-xs text-destructive">{health.error}</span>
            ) : null}
          </div>
        </section>

        <Tabs defaultValue="train" className="w-full">
          <TabsList className="grid w-full grid-cols-2 gap-2 p-1 md:grid-cols-4">
            <TabsTrigger value="train">Train</TabsTrigger>
            <TabsTrigger value="dataset">Dataset</TabsTrigger>
            <TabsTrigger value="predict">Predict</TabsTrigger>
            <TabsTrigger value="report">Latest Report</TabsTrigger>
          </TabsList>

          <TabsContent value="train" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Run Training</CardTitle>
                <CardDescription>
                  Trigger a new model training run and inspect metrics and generated artifacts.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleTrain} className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="test_size">Test Size</Label>
                    <Input
                      id="test_size"
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={trainForm.test_size}
                      onChange={(event) => updateTrainField('test_size', event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="random_state">Random State</Label>
                    <Input
                      id="random_state"
                      type="number"
                      value={trainForm.random_state}
                      onChange={(event) => updateTrainField('random_state', event.target.value)}
                    />
                  </div>
                  <div className="md:col-span-2 flex justify-end">
                    <Button type="submit" disabled={trainLoading}>
                      {trainLoading ? 'Training...' : 'Train Model'}
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>

            {trainError ? (
              <Alert variant="destructive">
                <AlertTitle>Training Failed</AlertTitle>
                <AlertDescription>{trainError}</AlertDescription>
              </Alert>
            ) : null}

            <section className="grid gap-4 md:grid-cols-5">
              {METRIC_KEYS.map((metric) => (
                <Card key={metric}>
                  <CardHeader className="pb-2">
                    <CardDescription>{prettyMetricName(metric)}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {trainLoading && !trainResult ? (
                      <Skeleton className="h-8 w-20" />
                    ) : (
                      <p className="text-xl font-semibold">
                        {trainMetrics?.[metric] ?? '--'}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </section>

            <Card>
              <CardHeader>
                <CardTitle>Training Metadata</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2">
                <div>
                  <span className="font-medium text-foreground">Run ID: </span>
                  {trainResult?.run_id ?? '--'}
                </div>
                <div>
                  <span className="font-medium text-foreground">Model Path: </span>
                  <PathValue value={trainResult?.model_path} />
                </div>
                <div>
                  <span className="font-medium text-foreground">Report Dir: </span>
                  <PathValue value={trainResult?.report_dir} />
                </div>
                <div>
                  <span className="font-medium text-foreground">Metrics Path: </span>
                  <PathValue value={trainResult?.metrics_path} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Graphs</CardTitle>
                <CardDescription>
                  All graph URLs returned from training are rendered below.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {trainLoading && !trainResult ? (
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {Array.from({ length: 3 }).map((_, index) => (
                      <Skeleton key={index} className="h-40 w-full" />
                    ))}
                  </div>
                ) : trainGraphs.length > 0 ? (
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {trainGraphs.map((graph) => (
                      <GraphCard
                        key={graph.id}
                        title={graph.title}
                        source={graph.source}
                        url={graph.url}
                        broken={brokenImages[graph.url]}
                        onBroken={markImageBroken}
                        onOpen={openGraphViewer}
                      />
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No graphs available yet. Run training to see plots.
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="dataset" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <CardTitle>Dataset Browser</CardTitle>
                    <CardDescription>
                      Fetch paginated dataset rows from the backend.
                    </CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => setDatasetInfoOpen(true)}>
                    Info
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3 md:grid-cols-[auto_auto] md:items-end">
                  <div className="space-y-2">
                    <Label>Page Size</Label>
                    <Select
                      value={datasetPageSize}
                      onValueChange={(value) => {
                        setDatasetPageSize(value)
                        setDatasetPage(1)
                      }}>
                      <SelectTrigger className="w-28">
                        <SelectValue placeholder="20" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="10">10</SelectItem>
                        <SelectItem value="20">20</SelectItem>
                        <SelectItem value="50">50</SelectItem>
                        <SelectItem value="100">100</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setDatasetPage(1)
                      handleFetchDataset()
                    }}
                    disabled={datasetLoading}>
                    Refresh
                  </Button>
                </div>

                {datasetError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Dataset Request Failed</AlertTitle>
                    <AlertDescription>{datasetError}</AlertDescription>
                  </Alert>
                ) : null}

                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="outline">
                    Total Records: {datasetPagination?.total_records ?? 0}
                  </Badge>
                  <Badge variant="outline">
                    Total Pages: {datasetPagination?.total_pages ?? 0}
                  </Badge>
                  <Badge variant="outline">
                    Current Page: {datasetPagination?.page ?? datasetPage}
                  </Badge>
                </div>

                <ScrollArea className="w-full rounded-md border">
                  <div className="min-w-180">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          {datasetColumns.map((column) => {
                            const description = getDatasetColumnDescription(column)

                            return (
                              <TableHead key={column}>
                                <div className="flex items-center gap-1.5">
                                  <span>{column}</span>
                                  {description ? (
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <button
                                          type="button"
                                          className="inline-flex items-center text-muted-foreground transition-colors hover:text-foreground"
                                          aria-label={`Info about ${column}`}>
                                          <Info className="size-3.5" />
                                        </button>
                                      </TooltipTrigger>
                                      <TooltipContent className="max-w-xs text-xs">
                                        {description}
                                      </TooltipContent>
                                    </Tooltip>
                                  ) : null}
                                </div>
                              </TableHead>
                            )
                          })}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {datasetLoading ? (
                          <TableRow>
                            <TableCell colSpan={Math.max(datasetColumns.length, 1)}>
                              <div className="space-y-2 py-2">
                                <Skeleton className="h-5 w-full" />
                                <Skeleton className="h-5 w-4/5" />
                                <Skeleton className="h-5 w-3/5" />
                              </div>
                            </TableCell>
                          </TableRow>
                        ) : datasetRows.length > 0 ? (
                          datasetRows.map((row, rowIndex) => (
                            <TableRow key={`row-${rowIndex}`}>
                              {datasetColumns.map((column) => (
                                <TableCell key={`${rowIndex}-${column}`}>
                                  {String(row?.[column] ?? '')}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))
                        ) : (
                          <TableRow>
                            <TableCell colSpan={Math.max(datasetColumns.length, 1)}>
                              No data available.
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </ScrollArea>

                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">
                    Dataset: <PathValue value={datasetResult?.dataset_path} />
                  </p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      onClick={() => setDatasetPage((value) => Math.max(1, value - 1))}
                      disabled={datasetLoading || !datasetPagination?.has_prev}>
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        setDatasetPage((value) =>
                          datasetPagination?.total_pages
                            ? Math.min(datasetPagination.total_pages, value + 1)
                            : value + 1,
                        )
                      }
                      disabled={datasetLoading || !datasetPagination?.has_next}>
                      Next
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="predict" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Prediction Model</CardTitle>
                <CardDescription>Select a trained model to start prediction.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
                  <div className="space-y-2">
                    <Label htmlFor="prediction-model">Model</Label>
                    <Select
                      value={selectedModelPath || undefined}
                      onValueChange={(value) => setSelectedModelPath(value)}
                      disabled={modelsLoading || !hasAnyModel}>
                      <SelectTrigger id="prediction-model">
                        <SelectValue placeholder={hasAnyModel ? 'Select a model' : 'No models'} />
                      </SelectTrigger>
                      <SelectContent>
                        {hasAnyModel ? (
                          modelsCatalog.models.map((model) => (
                            <SelectItem key={model.model_path} value={model.model_path}>
                              {getModelDisplayName(model)}
                            </SelectItem>
                          ))
                        ) : (
                          <SelectItem value={NO_MODELS_VALUE} disabled>
                            No models
                          </SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                    <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                      {selectedModelPath ? <Badge variant="outline">Custom model active</Badge> : null}
                      {activeModelPath ? (
                        <span>
                          Active path: <PathValue value={activeModelPath} />
                        </span>
                      ) : null}
                    </div>
                    {modelsError ? (
                      <Alert variant="destructive">
                        <AlertTitle>Model List Failed</AlertTitle>
                        <AlertDescription>{modelsError}</AlertDescription>
                      </Alert>
                    ) : null}
                  </div>

                  <div className="grid gap-3 sm:grid-cols-1">
                    <SummaryCard
                      label="Available Models"
                      value={modelsCatalog.model_count}
                      description={modelsLoading ? 'Loading model catalog...' : 'From /api/v1/models'}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Selected Model Details</CardTitle>
                <CardDescription>
                  Details refresh automatically when the model selection changes.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedModelLoading && !selectedModelDetails ? (
                  <div className="grid gap-3 md:grid-cols-2">
                    {Array.from({ length: 4 }).map((_, index) => (
                      <Skeleton key={index} className="h-24 w-full" />
                    ))}
                  </div>
                ) : selectedModelError ? (
                  <Alert>
                    <AlertTitle>Model Unavailable</AlertTitle>
                    <AlertDescription>{normalizeModelUiMessage(selectedModelError)}</AlertDescription>
                  </Alert>
                ) : activeModel ? (
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                      <SummaryCard label="Model Name" value={getModelDisplayName(activeModel)} />
                      <SummaryCard label="Size" value={formatBytes(activeModel.size_bytes)} />
                      <SummaryCard label="Has Report" value={activeModel.has_report ? 'Yes' : 'No'} />
                    </div>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2 rounded-lg border p-4">
                        <div className="text-sm font-medium">Paths</div>
                        <div className="space-y-2 text-sm text-muted-foreground">
                          <div>
                            <span className="font-medium text-foreground">Model Path: </span>
                            <PathValue value={activeModel.model_path} />
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Report Dir: </span>
                            <PathValue value={activeModel.report_dir} />
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Metrics Path: </span>
                            <PathValue value={activeModel.metrics_path} />
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Dataset Path: </span>
                            <PathValue value={activeModel.dataset_path} />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2 rounded-lg border p-4">
                        <div className="text-sm font-medium">Model Summary</div>
                        <div className="grid gap-2 text-sm text-muted-foreground">
                          <div>
                            <span className="font-medium text-foreground">Target: </span>
                            {activeModel.target || '--'}
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Test Size: </span>
                            {activeModel.test_size ?? '--'}
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Random State: </span>
                            {activeModel.random_state ?? '--'}
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Modified: </span>
                            {activeModel.modified_at || '--'}
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Created: </span>
                            {activeModel.created_at || '--'}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-3 md:grid-cols-5">
                      {METRIC_KEYS.map((metric) => (
                        <SummaryCard
                          key={metric}
                          label={prettyMetricName(metric)}
                          value={activeModel.metrics?.[metric] ?? '--'}
                        />
                      ))}
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Feature Names</div>
                      <div className="flex flex-wrap gap-2">
                        {(activeModel.feature_names ?? []).length > 0 ? (
                          activeModel.feature_names.map((feature) => (
                            <Badge key={feature} variant="outline">
                              {feature}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-sm text-muted-foreground">No feature metadata available.</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {hasAnyModel
                      ? 'Select a model to view details.'
                      : 'No models available yet, please train one to use that.'}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Predict</CardTitle>
                <CardDescription>
                  Paste a JSON object for one record or a JSON array for batch inference.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">Prediction mode: {selectedModelPath ? 'Custom model' : 'None selected'}</Badge>
                  {selectedModelPath ? (
                    <Badge variant="secondary">{getModelDisplayName(activeModel)}</Badge>
                  ) : null}
                </div>
                <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] xl:items-start">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="predict-json">Records JSON</Label>
                      <Textarea
                        id="predict-json"
                        value={predictInput}
                        onChange={(event) => setPredictInput(event.target.value)}
                        className="min-h-56 font-mono text-sm"
                      />
                    </div>

                    <div className="flex justify-end">
                      <Button onClick={handlePredict} disabled={predictLoading}>
                        {predictLoading ? 'Running...' : 'Run Prediction'}
                      </Button>
                    </div>

                    {predictError ? (
                      <Alert variant="destructive">
                        <AlertTitle>Prediction Failed</AlertTitle>
                        <AlertDescription>{predictError}</AlertDescription>
                      </Alert>
                    ) : null}
                  </div>

                  <div className="space-y-3 rounded-lg border p-4">
                    <div>
                      <p className="text-sm font-medium">Synthetic Data Generator</p>
                      <p className="text-xs text-muted-foreground">
                        Generate records and auto-fill the prediction editor.
                      </p>
                    </div>

                    <form
                      onSubmit={(event) => {
                        event.preventDefault()
                        handleGenerateSyntheticData()
                      }}
                      className="flex flex-wrap items-end gap-2">
                      <div className="flex-1 min-w-40">
                        <Label htmlFor="synthetic-count" className="text-xs mb-1 block">
                          <Hash className="inline size-3 mr-1 align-text-bottom" />
                          Batch
                        </Label>
                        <InputGroupInput
                          id="synthetic-count"
                          type="number"
                          min="1"
                          placeholder="10"
                          value={syntheticForm.count}
                          onChange={(event) => updateSyntheticField('count', event.target.value)}
                          className="h-10 text-center text-lg font-semibold"
                        />
                      </div>

                      <div className="flex-1 min-w-40">
                        <Label htmlFor="synthetic-random-state" className="text-xs mb-1 block">
                          <Shuffle className="inline size-3 mr-1 align-text-bottom" />
                          Seed
                        </Label>
                        <InputGroupInput
                          id="synthetic-random-state"
                          type="number"
                          placeholder="42"
                          value={syntheticForm.random_state}
                          onChange={(event) => updateSyntheticField('random_state', event.target.value)}
                          className="h-10 text-center text-lg font-semibold"
                        />
                      </div>

                      <Button 
                        type="submit" 
                        disabled={syntheticLoading}
                        className="h-10 min-w-32 rounded-lg bg-lime-500 px-4 text-base font-semibold text-black hover:bg-lime-400 disabled:bg-lime-500/70">
                        {syntheticLoading ? 'Generating...' : 'Generate'}
                      </Button>
                    </form>

                    {syntheticError ? (
                      <Alert variant="destructive">
                        <AlertTitle>Synthetic Data Failed</AlertTitle>
                        <AlertDescription>{syntheticError}</AlertDescription>
                      </Alert>
                    ) : null}

                    {syntheticLoading ? (
                      <div className="grid gap-2 sm:grid-cols-2">
                        {Array.from({ length: 2 }).map((_, index) => (
                          <Skeleton key={index} className="h-16 w-full" />
                        ))}
                      </div>
                    ) : syntheticResult ? (
                      <div className="space-y-2 text-xs text-muted-foreground">
                        <div className="grid gap-2 sm:grid-cols-2">
                          <div>
                            <span className="font-medium text-foreground">Requested:</span>{' '}
                            {syntheticResult.requested_count ?? '--'}
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Generated:</span>{' '}
                            {syntheticResult.actual_count ?? '--'}
                          </div>
                        </div>
                        <ScrollArea className="max-h-40 rounded-md border">
                          <pre className="whitespace-pre-wrap wrap-break-word p-3 text-xs">
                            {safeJsonStringify(syntheticResult.records) || 'No records returned.'}
                          </pre>
                        </ScrollArea>
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Fill batch and seed, then generate records for prediction.
                      </p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Prediction Results</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm font-medium">Required Features</p>
                  <div className="flex flex-wrap gap-2">
                    {(predictResult?.required_features ?? []).map((feature) => (
                      <Badge key={feature} variant="outline">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>

                <Separator />

                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Prediction</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(predictResult?.predictions ?? []).length > 0 ? (
                      predictResult.predictions.map((prediction, index) => (
                        <TableRow key={`pred-${index}`}>
                          <TableCell>{index + 1}</TableCell>
                          <TableCell>{prediction}</TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={2}>No predictions yet.</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

          </TabsContent>

          <TabsContent value="report" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Latest Report</CardTitle>
                <CardDescription>
                  Fetch and render the latest saved training report from the backend.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-end">
                  <Button onClick={handleFetchLatestReport} disabled={reportLoading}>
                    {reportLoading ? 'Fetching...' : 'Load Latest Report'}
                  </Button>
                </div>

                {reportError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Report Request Failed</AlertTitle>
                    <AlertDescription>{reportError}</AlertDescription>
                  </Alert>
                ) : null}

                <section className="grid gap-4 md:grid-cols-5">
                  {METRIC_KEYS.map((metric) => (
                    <Card key={`report-${metric}`}>
                      <CardHeader className="pb-2">
                        <CardDescription>{prettyMetricName(metric)}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-xl font-semibold">
                          {reportMetrics?.[metric] ?? '--'}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </section>

                <Card>
                  <CardHeader>
                    <CardTitle>Report Metadata</CardTitle>
                  </CardHeader>
                  <CardContent className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2">
                    <div>
                      <span className="font-medium text-foreground">Metrics File: </span>
                      <PathValue value={reportResult?.metrics_file} />
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Model Path: </span>
                      <PathValue value={reportResult?.model_path} />
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Dataset Path: </span>
                      <PathValue value={reportResult?.dataset_path} />
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Test Size: </span>
                      {reportResult?.test_size ?? '--'}
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Random State: </span>
                      {reportResult?.random_state ?? '--'}
                    </div>
                  </CardContent>
                </Card>

                {reportPlots.length > 0 ? (
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {reportPlots.map(([name, relativeUrl]) => {
                      const absolute = buildAbsoluteUrl(relativeUrl)
                      return (
                        <GraphCard
                          key={name}
                          title={name}
                          source="report.plot_urls"
                          url={absolute}
                          broken={brokenImages[absolute]}
                          onBroken={markImageBroken}
                          onOpen={openGraphViewer}
                        />
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No report plots loaded yet.
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Dialog open={Boolean(viewerGraph)} onOpenChange={(open) => !open && setViewerGraph(null)}>
          <DialogContent className="max-h-[90vh] w-[min(96vw,1200px)] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>{viewerGraph?.title ?? 'Graph Preview'}</DialogTitle>
              <DialogDescription>{viewerGraph?.source ?? ''}</DialogDescription>
            </DialogHeader>
            {viewerGraph ? (
              <div className="space-y-3">
                <img
                  src={viewerGraph.url}
                  alt={viewerGraph.title}
                  className="max-h-[75vh] w-full rounded-lg border object-contain bg-background"
                />
                <p className="break-all text-xs text-muted-foreground">
                  {viewerGraph.url}
                </p>
              </div>
            ) : null}
          </DialogContent>
        </Dialog>

        <Dialog open={datasetInfoOpen} onOpenChange={setDatasetInfoOpen}>
          <DialogContent className="max-h-[90vh] w-[min(96vw,980px)] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Dataset Information</DialogTitle>
              <DialogDescription>
                Column descriptions for the BigMart dataset used in this dashboard.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <ScrollArea className="max-h-[60vh] rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-64">Column</TableHead>
                      <TableHead>Description</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {DATASET_COLUMN_INFO.map((item) => (
                      <TableRow key={item.column}>
                        <TableCell className="font-medium">{item.column}</TableCell>
                        <TableCell>{item.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>

              <p className="text-sm text-muted-foreground">
                Source:{' '}
                <a
                  href={DATASET_SOURCE_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary underline-offset-4 hover:underline">
                  {DATASET_SOURCE_URL}
                </a>
              </p>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </main>
  )
}

export default App