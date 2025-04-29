import { useState, useEffect, useCallback } from 'react';
import { XCircleIcon, ArrowsPointingOutIcon } from '@heroicons/react/24/solid';
import { motion, AnimatePresence } from 'framer-motion';

const formatTime = (ms: number) => {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
};

const Timer = ({ startTime, endTime }: { startTime: number; endTime?: number }) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (endTime) {
      // If we have an end time, show final duration
      setElapsed(endTime - startTime);
      return;
    }

    // Only start interval if we don't have an end time
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, endTime]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute bottom-1.5 left-2 px-2 py-0.5 rounded-full bg-black/30 backdrop-blur-sm text-white text-xs font-mono"
      style={{ minWidth: '4rem', textAlign: 'center' }}
    >
      {formatTime(elapsed)}
    </motion.div>
  );
};

interface TaskUpdate {
  content: string;
  timestamp: number;
  type: 'screenshot' | 'thought';
}

interface TaskBoxProps {
  taskId: string;
  updates: TaskUpdate[];
  finalResult?: string;
  mainTaskFinalResult?: string;
  onClick?: () => void;
}

interface TaskModalProps {
  isOpen: boolean;
  onClose: () => void;
  updates: TaskUpdate[];
  finalResult?: string;
  mainTaskFinalResult?: string;
  taskId: string;
}

const getTaskTiming = (updates: TaskUpdate[], finalResult?: string) => {
  if (updates.length === 0) return null;

  const startTime = updates[0].timestamp;
  let endTime: number | undefined;

  // If we have a final result, find the last update's timestamp
  if (finalResult) {
    endTime = updates[updates.length - 1].timestamp;
  }

  return { startTime, endTime };
};

const findMatchingScreenshot = (thought: TaskUpdate, screenshots: TaskUpdate[], isLastThought: boolean): TaskUpdate | null => {
  // For the last thought, always show the most recent screenshot
  if (isLastThought && screenshots.length > 0) {
    return screenshots.reduce((latest, screenshot) => {
      if (!latest) return screenshot;
      return screenshot.timestamp > latest.timestamp ? screenshot : latest;
    }, null as TaskUpdate | null);
  }

  // Split screenshots into before and after the current thought
  const before = screenshots.filter(s => s.timestamp <= thought.timestamp);
  const after = screenshots.filter(s => s.timestamp > thought.timestamp);

  // If we have screenshots before the thought, use the most recent one
  if (before.length > 0) {
    return before.reduce((latest, screenshot) => {
      if (!latest) return screenshot;
      return screenshot.timestamp > latest.timestamp ? screenshot : latest;
    }, null as TaskUpdate | null);
  }

  // If no screenshots before, use the earliest one after
  if (after.length > 0) {
    return after.reduce((earliest, screenshot) => {
      if (!earliest) return screenshot;
      return screenshot.timestamp < earliest.timestamp ? screenshot : earliest;
    }, null as TaskUpdate | null);
  }

  return null;
};

const TaskModal = ({ isOpen, onClose, updates, finalResult, taskId }: TaskModalProps) => {
  // Get all thoughts
  const thoughts = updates.filter(u => u.type === 'thought');
  const screenshots = updates.filter(u => u.type === 'screenshot');

  // Always start with the last thought selected
  const [selectedIndex, setSelectedIndex] = useState(thoughts.length - 1);

  // Reset selection to last thought when modal opens
  useEffect(() => {
    if (isOpen) {
      setSelectedIndex(thoughts.length - 1);
    }
  }, [isOpen, thoughts.length]);

  // Find the appropriate screenshot for the selected thought
  const selectedThought = thoughts[selectedIndex];
  const currentScreenshot = selectedThought
    ? findMatchingScreenshot(
        selectedThought,
        screenshots,
        selectedIndex === thoughts.length - 1
      )
    : null;

  const timing = getTaskTiming(updates, finalResult);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
          transition={{ duration: 0.1 }}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="relative flex gap-6 max-w-5xl w-full mx-6"
            transition={{ duration: 0.1 }}
          >
            <button 
              onClick={onClose} 
              className="absolute -top-12 right-0 text-gray-400 hover:text-white transition-colors duration-[50ms]"
            >
              <XCircleIcon className="w-8 h-8" />
            </button>
            
            <div className="w-1/2">
              {currentScreenshot ? (
                <img
                  src={`data:image/png;base64,${currentScreenshot.content}`}
                  alt="Task Screenshot"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="pulse w-3 h-3 bg-white rounded-full" />
                </div>
              )}
            </div>
            
            <div className="w-1/2 max-h-[80vh] overflow-y-auto">
              {thoughts.map((thought, index) => (
                <motion.div
                  key={thought.timestamp}
                  onClick={() => setSelectedIndex(index)}
                  className={`p-4 cursor-pointer border-b border-gray-800 transition-colors duration-[100ms] ${
                    index === selectedIndex ? 'bg-[#262626]' : 'hover:bg-[#242424]'
                  }`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.1, delay: index * 0.05 }}
                >
                  <p className="text-sm text-gray-300">{thought.content}</p>
                  {index === thoughts.length - 1 && (
                    <p className="text-xs text-gray-500 mt-1">Last Result</p>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export const TaskBox = ({ taskId, updates, finalResult, mainTaskFinalResult, onClick }: TaskBoxProps) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  
  const latestScreenshot = updates
    .slice()
    .reverse()
    .find(u => u.type === 'screenshot')?.content;
    
  const latestThought = updates
    .slice()
    .reverse()
    .find(u => u.type === 'thought')?.content || '';

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="cursor-pointer"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        transition={{ duration: 0.1 }}
      >
        <div 
          className="aspect-video rounded-lg overflow-hidden mb-2 relative"
          onClick={() => setIsModalOpen(true)}
        >
          {latestScreenshot ? (
            <>
              <img
                src={`data:image/png;base64,${latestScreenshot}`}
                alt="Task Screenshot"
                className="w-full h-full object-cover"
              />
              <AnimatePresence>
                {isHovered && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute top-2 right-2 p-2 rounded-full bg-black/30 backdrop-blur-sm cursor-pointer hover:bg-black/40 transition-colors duration-[50ms]"
                  >
                    <ArrowsPointingOutIcon className="w-5 h-5 text-white" />
                  </motion.div>
                )}
              </AnimatePresence>
              {updates.length > 0 && (
                <Timer
                  startTime={updates[0].timestamp}
                  endTime={
                    // Stop timer when either:
                    // 1. This task has a final result OR
                    // 2. The main task has completed (indicated by mainTaskFinalResult)
                    (finalResult || mainTaskFinalResult)
                      ? updates[updates.length - 1].timestamp
                      : undefined
                  }
                />
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="pulse w-3 h-3 bg-white rounded-full" />
            </div>
          )}
        </div>
        <p className="text-sm text-gray-300 line-clamp-2">{latestThought}</p>
      </motion.div>

      <TaskModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        updates={updates}
        finalResult={finalResult}
        mainTaskFinalResult={mainTaskFinalResult}
        taskId={taskId}
      />
    </>
  );
};