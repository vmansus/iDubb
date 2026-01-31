import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Plus, Trash2, Upload, Download, Search, X, Edit2, Save } from 'lucide-react'
import { glossaryApi } from '../services/api'
import type { GlossaryEntry } from '../types'

export default function Glossary() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  // State
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [editingEntry, setEditingEntry] = useState<string | null>(null)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newEntry, setNewEntry] = useState({ source: '', target: '', note: '', category: 'general' })
  const [editForm, setEditForm] = useState({ source: '', target: '', note: '', category: '' })

  // Queries
  const { data: glossaryData, isLoading } = useQuery({
    queryKey: ['glossary', selectedCategory, search],
    queryFn: () => glossaryApi.getAll(selectedCategory || undefined, search || undefined),
  })

  // Mutations
  const addMutation = useMutation({
    mutationFn: glossaryApi.add,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['glossary'] })
      setShowAddForm(false)
      setNewEntry({ source: '', target: '', note: '', category: 'general' })
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ source, entry }: { source: string; entry: typeof editForm }) =>
      glossaryApi.update(source, entry),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['glossary'] })
      setEditingEntry(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: glossaryApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['glossary'] })
    },
  })

  const clearMutation = useMutation({
    mutationFn: glossaryApi.clear,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['glossary'] })
    },
  })

  // Handlers
  const handleAdd = () => {
    if (newEntry.source && newEntry.target) {
      addMutation.mutate(newEntry)
    }
  }

  const handleEdit = (entry: GlossaryEntry) => {
    setEditingEntry(entry.source)
    setEditForm({
      source: entry.source,
      target: entry.target,
      note: entry.note,
      category: entry.category,
    })
  }

  const handleSaveEdit = () => {
    if (editingEntry && editForm.source && editForm.target) {
      updateMutation.mutate({ source: editingEntry, entry: editForm })
    }
  }

  const handleDelete = (source: string) => {
    if (confirm(t('glossary.confirmDelete'))) {
      deleteMutation.mutate(source)
    }
  }

  const handleClear = () => {
    if (confirm(t('glossary.confirmClear'))) {
      clearMutation.mutate()
    }
  }

  const handleExport = async () => {
    const data = await glossaryApi.exportJson()
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'glossary.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const text = await file.text()
    try {
      if (file.name.endsWith('.csv')) {
        await glossaryApi.import('csv', text)
      } else {
        await glossaryApi.import('json', text)
      }
      queryClient.invalidateQueries({ queryKey: ['glossary'] })
    } catch (error) {
      console.error('Import failed:', error)
    }
    e.target.value = ''
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{t('glossary.title')}</h1>
          <p className="text-sm text-gray-500 mt-1">{t('glossary.description')}</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 cursor-pointer">
            <Upload className="w-4 h-4" />
            {t('glossary.import')}
            <input
              type="file"
              accept=".json,.csv"
              onChange={handleImport}
              className="hidden"
            />
          </label>
          <button
            onClick={handleExport}
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            <Download className="w-4 h-4" />
            {t('glossary.export')}
          </button>
          <button
            onClick={() => setShowAddForm(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Plus className="w-4 h-4" />
            {t('glossary.add')}
          </button>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('glossary.searchPlaceholder')}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        {glossaryData?.categories && glossaryData.categories.length > 0 && (
          <select
            value={selectedCategory || ''}
            onChange={(e) => setSelectedCategory(e.target.value || null)}
            className="px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option value="">{t('glossary.allCategories')}</option>
            {glossaryData.categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        )}
      </div>

      {/* Add Form */}
      {showAddForm && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-3">{t('glossary.addNew')}</h3>
          <div className="grid grid-cols-4 gap-3">
            <input
              type="text"
              value={newEntry.source}
              onChange={(e) => setNewEntry({ ...newEntry, source: e.target.value })}
              placeholder={t('glossary.sourceTerm')}
              className="px-3 py-2 border border-gray-300 rounded-md"
            />
            <input
              type="text"
              value={newEntry.target}
              onChange={(e) => setNewEntry({ ...newEntry, target: e.target.value })}
              placeholder={t('glossary.targetTerm')}
              className="px-3 py-2 border border-gray-300 rounded-md"
            />
            <input
              type="text"
              value={newEntry.note}
              onChange={(e) => setNewEntry({ ...newEntry, note: e.target.value })}
              placeholder={t('glossary.note')}
              className="px-3 py-2 border border-gray-300 rounded-md"
            />
            <div className="flex gap-2">
              <input
                type="text"
                value={newEntry.category}
                onChange={(e) => setNewEntry({ ...newEntry, category: e.target.value })}
                placeholder={t('glossary.category')}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
              />
              <button
                onClick={handleAdd}
                disabled={!newEntry.source || !newEntry.target}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                <Save className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Entries Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{t('glossary.sourceTerm')}</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{t('glossary.targetTerm')}</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{t('glossary.note')}</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{t('glossary.category')}</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">{t('glossary.actions')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {isLoading ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  {t('common.loading')}
                </td>
              </tr>
            ) : glossaryData?.entries.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  {t('glossary.empty')}
                </td>
              </tr>
            ) : (
              glossaryData?.entries.map((entry) => (
                <tr key={entry.source} className="hover:bg-gray-50">
                  {editingEntry === entry.source ? (
                    <>
                      <td className="px-4 py-2">
                        <input
                          type="text"
                          value={editForm.source}
                          onChange={(e) => setEditForm({ ...editForm, source: e.target.value })}
                          className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        />
                      </td>
                      <td className="px-4 py-2">
                        <input
                          type="text"
                          value={editForm.target}
                          onChange={(e) => setEditForm({ ...editForm, target: e.target.value })}
                          className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        />
                      </td>
                      <td className="px-4 py-2">
                        <input
                          type="text"
                          value={editForm.note}
                          onChange={(e) => setEditForm({ ...editForm, note: e.target.value })}
                          className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        />
                      </td>
                      <td className="px-4 py-2">
                        <input
                          type="text"
                          value={editForm.category}
                          onChange={(e) => setEditForm({ ...editForm, category: e.target.value })}
                          className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        />
                      </td>
                      <td className="px-4 py-2 text-right">
                        <button
                          onClick={handleSaveEdit}
                          className="text-green-600 hover:text-green-700 p-1"
                        >
                          <Save className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setEditingEntry(null)}
                          className="text-gray-600 hover:text-gray-700 p-1 ml-1"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="px-4 py-3 text-sm font-medium text-gray-900">{entry.source}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{entry.target}</td>
                      <td className="px-4 py-3 text-sm text-gray-500">{entry.note}</td>
                      <td className="px-4 py-3">
                        <span className="inline-flex px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 rounded">
                          {entry.category}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleEdit(entry)}
                          className="text-blue-600 hover:text-blue-700 p-1"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(entry.source)}
                          className="text-red-600 hover:text-red-700 p-1 ml-1"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </>
                  )}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>{t('glossary.totalEntries', { count: glossaryData?.total || 0 })}</span>
        {(glossaryData?.total || 0) > 0 && (
          <button
            onClick={handleClear}
            className="text-red-600 hover:text-red-700"
          >
            {t('glossary.clearAll')}
          </button>
        )}
      </div>
    </div>
  )
}
