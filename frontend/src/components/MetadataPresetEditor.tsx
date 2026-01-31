import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Plus, Tag } from 'lucide-react'
import type { MetadataPreset, CreateMetadataPresetRequest } from '../types'

interface MetadataPresetEditorProps {
  preset?: MetadataPreset | null  // null = create new
  onSave: (data: CreateMetadataPresetRequest) => void
  onCancel: () => void
  isSaving?: boolean
}

const SUGGESTED_TAGS = [
  '通用', '字幕', '翻译', '双语', '科技', '教程', 'AI', '技术',
  '娱乐', '休闲', '综艺', '游戏', '自制', '原创',
]

export default function MetadataPresetEditor({
  preset,
  onSave,
  onCancel,
  isSaving = false,
}: MetadataPresetEditorProps) {
  const { t } = useTranslation()
  const isEditing = !!preset

  const [formData, setFormData] = useState<CreateMetadataPresetRequest>({
    name: '',
    description: '',
    title_prefix: '',
    custom_signature: '',
    tags: [],
  })
  const [newTag, setNewTag] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    if (preset) {
      setFormData({
        name: preset.name,
        description: preset.description || '',
        title_prefix: preset.title_prefix,
        custom_signature: preset.custom_signature,
        tags: preset.tags || [],
      })
    }
  }, [preset])

  const handleChange = (field: keyof CreateMetadataPresetRequest, value: string | string[]) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = { ...prev }
        delete newErrors[field]
        return newErrors
      })
    }
  }

  const addTag = (tag: string) => {
    const trimmedTag = tag.trim()
    if (trimmedTag && !formData.tags.includes(trimmedTag)) {
      handleChange('tags', [...formData.tags, trimmedTag])
    }
    setNewTag('')
  }

  const removeTag = (tagToRemove: string) => {
    handleChange('tags', formData.tags.filter(tag => tag !== tagToRemove))
  }

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.name.trim()) {
      newErrors.name = t('metadataPreset.errors.nameRequired')
    } else if (formData.name.length > 100) {
      newErrors.name = t('metadataPreset.errors.nameTooLong')
    }

    if (formData.title_prefix.length > 50) {
      newErrors.title_prefix = t('metadataPreset.errors.prefixTooLong')
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (validate()) {
      onSave(formData)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">
            {isEditing
              ? t('metadataPreset.editTitle')
              : t('metadataPreset.createTitle')}
          </h2>
          <button
            onClick={onCancel}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('metadataPreset.name')} <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              placeholder={t('metadataPreset.namePlaceholder')}
              className={`w-full px-3 py-2 border rounded-md text-sm ${
                errors.name ? 'border-red-500' : 'border-gray-300'
              }`}
              maxLength={100}
            />
            {errors.name && (
              <p className="mt-1 text-xs text-red-500">{errors.name}</p>
            )}
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('metadataPreset.description')}
            </label>
            <input
              type="text"
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              placeholder={t('metadataPreset.descriptionPlaceholder')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
          </div>

          {/* Title Prefix */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('metadataPreset.titlePrefix')}
            </label>
            <input
              type="text"
              value={formData.title_prefix}
              onChange={(e) => handleChange('title_prefix', e.target.value)}
              placeholder={t('metadataPreset.prefixPlaceholder')}
              className={`w-full px-3 py-2 border rounded-md text-sm font-mono ${
                errors.title_prefix ? 'border-red-500' : 'border-gray-300'
              }`}
              maxLength={50}
            />
            {errors.title_prefix && (
              <p className="mt-1 text-xs text-red-500">{errors.title_prefix}</p>
            )}
            <p className="mt-1 text-xs text-gray-500">
              {t('metadataPreset.prefixHelp')}
            </p>
          </div>

          {/* Custom Signature */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('metadataPreset.customSignature')}
            </label>
            <textarea
              value={formData.custom_signature}
              onChange={(e) => handleChange('custom_signature', e.target.value)}
              placeholder={t('metadataPreset.signaturePlaceholder')}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">
              {t('metadataPreset.signatureHelp')}
            </p>
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('metadataPreset.tags')}
            </label>

            {/* Current Tags */}
            {formData.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {formData.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-1 px-2 py-1 bg-blue-50 text-blue-700 rounded-full text-xs"
                  >
                    <Tag className="w-3 h-3" />
                    {tag}
                    <button
                      type="button"
                      onClick={() => removeTag(tag)}
                      className="hover:text-blue-900"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}

            {/* Add Tag Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    addTag(newTag)
                  }
                }}
                placeholder={t('metadataPreset.addTagPlaceholder')}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
              />
              <button
                type="button"
                onClick={() => addTag(newTag)}
                className="px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {/* Suggested Tags */}
            <div className="mt-2">
              <p className="text-xs text-gray-500 mb-1">{t('metadataPreset.suggestedTags')}:</p>
              <div className="flex flex-wrap gap-1">
                {SUGGESTED_TAGS.filter(tag => !formData.tags.includes(tag)).slice(0, 8).map((tag, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => addTag(tag)}
                    className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full text-xs hover:bg-gray-200 transition-colors"
                  >
                    + {tag}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Preview */}
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-xs font-medium text-gray-500 mb-2">{t('metadataPreset.preview')}</p>
            <p className="text-sm text-gray-700">
              {formData.title_prefix && (
                <span className="font-mono text-blue-600">{formData.title_prefix} </span>
              )}
              <span className="text-gray-400">{t('metadataPreset.sampleTitle')}</span>
            </p>
            {formData.custom_signature && (
              <p className="text-xs text-gray-500 mt-1 border-t pt-1">
                {formData.custom_signature.substring(0, 100)}
                {formData.custom_signature.length > 100 && '...'}
              </p>
            )}
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t bg-gray-50">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded-md transition-colors"
          >
            {t('common.cancel')}
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSaving}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isSaving ? t('common.saving') : t('common.save')}
          </button>
        </div>
      </div>
    </div>
  )
}
