import { useState, useEffect, useCallback, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { QrCode, Trash2, Loader2, CheckCircle, User, RefreshCw, Star, Edit2, Check, X } from 'lucide-react'
import { bilibiliApi, type BilibiliAccount } from '../services/api'
import QRCode from 'qrcode'

interface QRLoginModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

function QRLoginModal({ isOpen, onClose, onSuccess }: QRLoginModalProps) {
  const [qrcodeKey, setQrcodeKey] = useState<string | null>(null)
  const [qrcodeDataUrl, setQrcodeDataUrl] = useState<string | null>(null)
  const [status, setStatus] = useState<'loading' | 'waiting' | 'scanned' | 'success' | 'expired' | 'error'>('loading')
  const [message, setMessage] = useState('')
  const [label, setLabel] = useState('')
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Generate QR code
  const generateQR = useCallback(async () => {
    setStatus('loading')
    setMessage('')
    try {
      const data = await bilibiliApi.generateQRCode()
      setQrcodeKey(data.qrcode_key)
      
      // Generate QR code image
      const dataUrl = await QRCode.toDataURL(data.qrcode_url, {
        width: 256,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#ffffff',
        },
      })
      setQrcodeDataUrl(dataUrl)
      setStatus('waiting')
    } catch (error) {
      setStatus('error')
      setMessage('生成二维码失败，请重试')
    }
  }, [])

  // Stop polling helper
  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
  }, [])

  // Poll for scan status
  const pollStatus = useCallback(async () => {
    if (!qrcodeKey) return

    try {
      const result = await bilibiliApi.pollQRCode(qrcodeKey, label || undefined)
      
      if (result.status === 'success') {
        setStatus('success')
        setMessage(result.message)
        stopPolling()
        // Notify parent
        setTimeout(() => {
          onSuccess()
          onClose()
        }, 1500)
      } else if (result.status === 'scanned') {
        setStatus('scanned')
        setMessage(result.message)
      } else if (result.status === 'expired') {
        setStatus('expired')
        setMessage(result.message)
        stopPolling()
      } else if (result.status === 'error') {
        setStatus('error')
        setMessage(result.message)
      }
    } catch (error) {
      console.error('Poll error:', error)
    }
  }, [qrcodeKey, label, stopPolling, onSuccess, onClose])

  // Start QR generation when modal opens
  useEffect(() => {
    if (isOpen) {
      setLabel('')
      generateQR()
    }
    return () => {
      stopPolling()
    }
  }, [isOpen, generateQR, stopPolling])

  // Start polling when QR code is ready
  useEffect(() => {
    if (status === 'waiting' || status === 'scanned') {
      stopPolling() // Clear any existing interval
      pollIntervalRef.current = setInterval(pollStatus, 2000)
    }
    return () => {
      stopPolling()
    }
  }, [status, pollStatus, stopPolling])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-sm shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">B站扫码登录</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        </div>

        {/* Label input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            账号标签（选填）
          </label>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="例如：游戏号、生活号"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-pink-500 focus:border-pink-500"
            disabled={status === 'scanned' || status === 'success'}
          />
          <p className="text-xs text-gray-500 mt-1">标签用于区分多个账号，留空则使用昵称</p>
        </div>

        <div className="flex flex-col items-center">
          {status === 'loading' && (
            <div className="w-64 h-64 flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-pink-500 animate-spin" />
            </div>
          )}

          {(status === 'waiting' || status === 'scanned') && qrcodeDataUrl && (
            <div className="relative">
              <img
                src={qrcodeDataUrl}
                alt="QR Code"
                className="w-64 h-64 rounded-lg border border-gray-200"
              />
              {status === 'scanned' && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
                  <div className="text-center">
                    <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                    <p className="text-white">请在手机上确认</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {status === 'success' && (
            <div className="w-64 h-64 flex flex-col items-center justify-center">
              <CheckCircle className="w-16 h-16 text-green-500 mb-4" />
              <p className="text-green-600 text-lg">{message}</p>
            </div>
          )}

          {status === 'expired' && (
            <div className="w-64 h-64 flex flex-col items-center justify-center">
              <p className="text-yellow-600 mb-4">{message}</p>
              <button
                onClick={generateQR}
                className="flex items-center gap-2 px-4 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600"
              >
                <RefreshCw className="w-4 h-4" />
                重新生成
              </button>
            </div>
          )}

          {status === 'error' && (
            <div className="w-64 h-64 flex flex-col items-center justify-center">
              <p className="text-red-600 mb-4">{message}</p>
              <button
                onClick={generateQR}
                className="flex items-center gap-2 px-4 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600"
              >
                <RefreshCw className="w-4 h-4" />
                重试
              </button>
            </div>
          )}
        </div>

        <p className="text-center text-gray-500 text-sm mt-4">
          请使用哔哩哔哩 App 扫描二维码登录
        </p>
      </div>
    </div>
  )
}

interface AccountItemProps {
  account: BilibiliAccount
  onDelete: (uid: string, label: string) => void
  onSetPrimary: (uid: string) => void
  onUpdateLabel: (uid: string, label: string) => void
  isDeleting: boolean
}

function AccountItem({ account, onDelete, onSetPrimary, onUpdateLabel, isDeleting }: AccountItemProps) {
  const [isEditingLabel, setIsEditingLabel] = useState(false)
  const [editLabel, setEditLabel] = useState(account.label)

  const handleSaveLabel = () => {
    if (editLabel.trim() && editLabel !== account.label) {
      onUpdateLabel(account.uid, editLabel.trim())
    }
    setIsEditingLabel(false)
  }

  const handleCancelEdit = () => {
    setEditLabel(account.label)
    setIsEditingLabel(false)
  }

  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex items-center gap-3">
        {account.avatar ? (
          <img
            src={account.avatar}
            alt={account.nickname}
            className="w-10 h-10 rounded-full"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center">
            <User className="w-5 h-5 text-gray-400" />
          </div>
        )}
        <div>
          <div className="flex items-center gap-2">
            {isEditingLabel ? (
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  value={editLabel}
                  onChange={(e) => setEditLabel(e.target.value)}
                  className="px-2 py-0.5 text-sm border border-gray-300 rounded w-32"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSaveLabel()
                    if (e.key === 'Escape') handleCancelEdit()
                  }}
                />
                <button onClick={handleSaveLabel} className="p-1 text-green-600 hover:bg-green-50 rounded">
                  <Check className="w-4 h-4" />
                </button>
                <button onClick={handleCancelEdit} className="p-1 text-gray-400 hover:bg-gray-100 rounded">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <span className="text-gray-900 font-medium">{account.label || account.nickname}</span>
                <button
                  onClick={() => setIsEditingLabel(true)}
                  className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
                  title="编辑标签"
                >
                  <Edit2 className="w-3 h-3" />
                </button>
                {account.is_primary && (
                  <span className="px-1.5 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">主账号</span>
                )}
              </>
            )}
          </div>
          <p className="text-gray-500 text-sm">
            {account.label ? account.nickname + ' · ' : ''}UID: {account.uid}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-1">
        {!account.is_primary && (
          <button
            onClick={() => onSetPrimary(account.uid)}
            className="p-2 text-gray-400 hover:text-yellow-500 hover:bg-gray-100 rounded-lg transition-colors"
            title="设为主账号"
          >
            <Star className="w-4 h-4" />
          </button>
        )}
        <button
          onClick={() => onDelete(account.uid, account.label || account.nickname)}
          disabled={isDeleting}
          className="p-2 text-gray-400 hover:text-red-500 hover:bg-gray-100 rounded-lg transition-colors"
          title="删除账号"
        >
          {isDeleting ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  )
}

export default function BilibiliAccounts() {
  const queryClient = useQueryClient()
  const [showQRModal, setShowQRModal] = useState(false)

  // Fetch accounts
  const { data: accountsData, isLoading } = useQuery({
    queryKey: ['bilibiliAccounts'],
    queryFn: bilibiliApi.listAccounts,
  })

  // Delete account mutation
  const deleteMutation = useMutation({
    mutationFn: bilibiliApi.deleteAccount,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bilibiliAccounts'] })
    },
  })

  // Set primary mutation
  const setPrimaryMutation = useMutation({
    mutationFn: bilibiliApi.setPrimary,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bilibiliAccounts'] })
    },
  })

  // Update label mutation
  const updateLabelMutation = useMutation({
    mutationFn: ({ uid, label }: { uid: string; label: string }) => 
      bilibiliApi.updateLabel(uid, label),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bilibiliAccounts'] })
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || '更新标签失败')
    },
  })

  const handleDelete = (uid: string, label: string) => {
    if (confirm(`确定要删除账号 "${label}" 吗？`)) {
      deleteMutation.mutate(uid)
    }
  }

  const handleSetPrimary = (uid: string) => {
    setPrimaryMutation.mutate(uid)
  }

  const handleUpdateLabel = (uid: string, label: string) => {
    updateLabelMutation.mutate({ uid, label })
  }

  const handleQRSuccess = () => {
    queryClient.invalidateQueries({ queryKey: ['bilibiliAccounts'] })
  }

  const accounts = accountsData?.accounts || []

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900 flex items-center gap-2">
          <User className="w-5 h-5 text-pink-500" />
          B站账号管理
        </h3>
        <button
          onClick={() => setShowQRModal(true)}
          className="flex items-center gap-2 px-3 py-1.5 bg-pink-500 text-white text-sm rounded-lg hover:bg-pink-600 transition-colors"
        >
          <QrCode className="w-4 h-4" />
          扫码添加
        </button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 text-pink-500 animate-spin" />
        </div>
      ) : accounts.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          <User className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>暂无已登录账号</p>
          <p className="text-sm">点击"扫码添加"登录B站账号</p>
        </div>
      ) : (
        <div className="space-y-2">
          {accounts.map((account) => (
            <AccountItem
              key={account.uid}
              account={account}
              onDelete={handleDelete}
              onSetPrimary={handleSetPrimary}
              onUpdateLabel={handleUpdateLabel}
              isDeleting={deleteMutation.isPending}
            />
          ))}
        </div>
      )}

      <p className="text-gray-500 text-sm">
        提示：标签用于快速识别账号，第一个添加的账号自动设为主账号
      </p>

      <QRLoginModal
        isOpen={showQRModal}
        onClose={() => setShowQRModal(false)}
        onSuccess={handleQRSuccess}
      />
    </div>
  )
}
