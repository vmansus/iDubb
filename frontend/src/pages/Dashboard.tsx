import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { PlusCircle, RefreshCw, Trash2, X, CheckSquare, Square, Loader2, FolderOpen } from 'lucide-react'
import { taskApi, directoryApi } from '../services/api'
import TaskCard from '../components/TaskCard'

export default function Dashboard() {
  const { t } = useTranslation()
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set())
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [taskToDelete, setTaskToDelete] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [selectedDirectory, setSelectedDirectory] = useState<string>('')

  // Fetch directories for filter
  const { data: directoriesData } = useQuery({
    queryKey: ['directories'],
    queryFn: () => directoryApi.list(),
    staleTime: 60000, // 1 minute
  })

  const { data: tasks, isLoading, refetch } = useQuery({
    queryKey: ['tasks', selectedDirectory],
    queryFn: () => taskApi.list(undefined, selectedDirectory || undefined),
    refetchInterval: 3000,
  })

  // Has any selection
  const hasSelection = selectedTasks.size > 0

  // Toggle task selection
  const toggleTaskSelection = (taskId: string) => {
    const newSelected = new Set(selectedTasks)
    if (newSelected.has(taskId)) {
      newSelected.delete(taskId)
    } else {
      newSelected.add(taskId)
    }
    setSelectedTasks(newSelected)
  }

  // Select all tasks
  const selectAllTasks = () => {
    if (tasks) {
      setSelectedTasks(new Set(tasks.map(t => t.task_id)))
    }
  }

  // Clear selection
  const clearSelection = () => {
    setSelectedTasks(new Set())
  }

  // Handle single task delete request (from TaskCard)
  const handleDeleteRequest = (taskId: string) => {
    setTaskToDelete(taskId)
    setShowDeleteConfirm(true)
  }

  // Handle batch delete request
  const handleBatchDeleteRequest = () => {
    if (selectedTasks.size > 0) {
      setTaskToDelete(null) // null means batch delete
      setShowDeleteConfirm(true)
    }
  }

  // Confirm and execute delete
  const confirmDelete = async () => {
    setIsDeleting(true)
    try {
      const tasksToDelete = taskToDelete ? [taskToDelete] : Array.from(selectedTasks)

      // Delete tasks in parallel
      await Promise.all(
        tasksToDelete.map(taskId =>
          fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8888'}/api/tasks/${taskId}`, {
            method: 'DELETE',
          })
        )
      )

      // Clear selection and close dialog
      setSelectedTasks(new Set())
      setShowDeleteConfirm(false)
      setTaskToDelete(null)
      refetch()
    } catch (error) {
      console.error('Failed to delete tasks:', error)
    } finally {
      setIsDeleting(false)
    }
  }

  const deleteCount = taskToDelete ? 1 : selectedTasks.size

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold text-gray-900">{t('dashboard.title')}</h2>
          {/* Directory Filter */}
          {directoriesData && directoriesData.directories.length > 0 && (
            <div className="relative">
              <select
                value={selectedDirectory}
                onChange={(e) => {
                  setSelectedDirectory(e.target.value)
                  setSelectedTasks(new Set()) // Clear selection when filter changes
                }}
                className="appearance-none pl-8 pr-8 py-1.5 text-sm border border-gray-300 rounded-md bg-white text-gray-700 hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
              >
                <option value="">{t('dashboard.allDirectories')}</option>
                {directoriesData.directories.map((dir) => (
                  <option key={dir.id} value={dir.name}>
                    {dir.name} ({dir.task_count})
                  </option>
                ))}
              </select>
              <FolderOpen className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
              <svg className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-3">
          {/* Batch action buttons - show when tasks are selected */}
          {hasSelection && (
            <>
              <span className="text-sm text-gray-500">
                {t('dashboard.selected', { count: selectedTasks.size })}
              </span>
              <button
                onClick={() => {
                  if (tasks && selectedTasks.size === tasks.length) {
                    setSelectedTasks(new Set())
                  } else {
                    selectAllTasks()
                  }
                }}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                {tasks && selectedTasks.size === tasks.length ? (
                  <>
                    <Square className="h-4 w-4 mr-2" />
                    {t('dashboard.deselectAll')}
                  </>
                ) : (
                  <>
                    <CheckSquare className="h-4 w-4 mr-2" />
                    {t('dashboard.selectAll')}
                  </>
                )}
              </button>
              <button
                onClick={handleBatchDeleteRequest}
                className="inline-flex items-center px-3 py-2 border border-red-300 rounded-md text-sm font-medium text-red-700 bg-red-50 hover:bg-red-100"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                {t('dashboard.deleteSelected')}
              </button>
              <button
                onClick={clearSelection}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <X className="h-4 w-4 mr-2" />
                {t('common.cancel')}
              </button>
            </>
          )}
          {/* Normal buttons */}
          {!hasSelection && (
            <>
              <button
                onClick={() => refetch()}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                {t('common.refresh')}
              </button>
              <Link
                to="/new"
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
              >
                <PlusCircle className="h-4 w-4 mr-2" />
                {t('nav.newTask')}
              </Link>
            </>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      ) : tasks && tasks.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {tasks.map((task) => (
            <TaskCard
              key={task.task_id}
              task={task}
              isSelected={selectedTasks.has(task.task_id)}
              onSelect={() => toggleTaskSelection(task.task_id)}
              onDeleteRequest={() => handleDeleteRequest(task.task_id)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <div className="text-gray-500 mb-4">{t('dashboard.noTasks')}</div>
          <Link
            to="/new"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
          >
            <PlusCircle className="h-4 w-4 mr-2" />
            {t('dashboard.createFirst')}
          </Link>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => !isDeleting && setShowDeleteConfirm(false)}
          />
          {/* Modal */}
          <div className="relative bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {t('dashboard.confirmDelete')}
            </h3>
            <p className="text-gray-600 mb-6">
              {t('dashboard.confirmDeleteMessage', { count: deleteCount })}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                disabled={isDeleting}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={confirmDelete}
                disabled={isDeleting}
                className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-md text-sm font-medium hover:bg-red-700 disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    {t('common.deleting')}
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-2" />
                    {t('common.delete')}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
