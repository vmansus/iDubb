import { useTranslation } from 'react-i18next'
import { Star, Edit2, Trash2, Tag, Lock } from 'lucide-react'
import type { MetadataPreset } from '../types'

interface MetadataPresetCardProps {
  preset: MetadataPreset
  onEdit?: () => void
  onDelete?: () => void
  onSetDefault?: () => void
  isDeleting?: boolean
}

export default function MetadataPresetCard({
  preset,
  onEdit,
  onDelete,
  onSetDefault,
  isDeleting = false,
}: MetadataPresetCardProps) {
  const { t } = useTranslation()

  return (
    <div
      className={`relative bg-white rounded-lg border-2 p-4 transition-all ${
        preset.is_default
          ? 'border-blue-500 shadow-md'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      {/* Default Badge */}
      {preset.is_default && (
        <div className="absolute -top-2 -right-2 bg-blue-500 text-white px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1">
          <Star className="w-3 h-3" />
          {t('metadataPreset.default')}
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="font-medium text-gray-900">{preset.name}</h3>
          {preset.is_builtin && (
            <span className="px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded text-xs flex items-center gap-1">
              <Lock className="w-3 h-3" />
              {t('metadataPreset.builtin')}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {!preset.is_default && (
            <button
              onClick={onSetDefault}
              className="p-1.5 text-gray-400 hover:text-blue-500 hover:bg-blue-50 rounded transition-colors"
              title={t('metadataPreset.setDefault')}
            >
              <Star className="w-4 h-4" />
            </button>
          )}
          {!preset.is_builtin && (
            <>
              <button
                onClick={onEdit}
                className="p-1.5 text-gray-400 hover:text-blue-500 hover:bg-blue-50 rounded transition-colors"
                title={t('common.edit')}
              >
                <Edit2 className="w-4 h-4" />
              </button>
              <button
                onClick={onDelete}
                disabled={isDeleting}
                className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                title={t('common.delete')}
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Description */}
      {preset.description && (
        <p className="text-sm text-gray-500 mb-3 line-clamp-2">
          {preset.description}
        </p>
      )}

      {/* Preview Section */}
      <div className="space-y-2 mb-3">
        {/* Title Prefix Preview */}
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-400 w-16 shrink-0">{t('metadataPreset.prefix')}:</span>
          {preset.title_prefix ? (
            <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded font-mono text-xs">
              {preset.title_prefix}
            </span>
          ) : (
            <span className="text-gray-300 text-xs">{t('metadataPreset.noPrefix')}</span>
          )}
        </div>

        {/* Signature Preview */}
        <div className="flex items-start gap-2 text-sm">
          <span className="text-gray-400 w-16 shrink-0">{t('metadataPreset.signature')}:</span>
          {preset.custom_signature ? (
            <span className="text-gray-600 text-xs line-clamp-2">
              {preset.custom_signature.substring(0, 80)}
              {preset.custom_signature.length > 80 && '...'}
            </span>
          ) : (
            <span className="text-gray-300 text-xs">{t('metadataPreset.noSignature')}</span>
          )}
        </div>
      </div>

      {/* Tags */}
      {preset.tags && preset.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {preset.tags.slice(0, 4).map((tag, index) => (
            <span
              key={index}
              className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full text-xs"
            >
              <Tag className="w-3 h-3" />
              {tag}
            </span>
          ))}
          {preset.tags.length > 4 && (
            <span className="px-2 py-0.5 text-gray-400 text-xs">
              +{preset.tags.length - 4}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
