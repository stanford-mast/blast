import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';
import OpenAI from 'openai';
import { Header } from '../components/Header';
import { ChatInput } from '../components/ChatInput';
import { TaskBox } from '../components/TaskBox';
import { LoadingLogo } from '../components/LoadingLogo';

// Initialize OpenAI client with custom base URL
// Get server port from environment variable or use default
const serverPort = process.env.NEXT_PUBLIC_SERVER_PORT || '8000';

const openai = new OpenAI({
  baseURL: `http://localhost:${serverPort}`,
  apiKey: 'not-needed',
  dangerouslyAllowBrowser: true,
});

interface TaskUpdate {
  content: string;
  timestamp: number;
  type: 'screenshot' | 'thought';
}

interface TaskState {
  taskId: string;
  updates: TaskUpdate[];
  finalResult?: string;
}

interface ConversationItem {
  type: 'user' | 'tasks';
  content: string;
  tasks?: Record<string, TaskState>;
  finalResult?: string;
  responseId?: string;
  showLoading?: boolean;
  hasFirstResponse?: boolean;
}

interface StreamEvent {
  type: string;
  item_id?: string;
  delta?: string;
  item?: {
    id: string;
    content?: Array<{
      type: string;
      text: string;
    }>;
  };
  response?: {
    id: string;
  };
}

// Helper function to check if a task has screenshots
const hasScreenshots = (task: TaskState) => {
  return task.updates.some(update => update.type === 'screenshot');
};

export default function Home() {
  const [showLogo, setShowLogo] = useState(true);
  const [showApp, setShowApp] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const firstTaskId = useRef<string | null>(null);
  const isFirstTaskDone = useRef<boolean>(false);

  useEffect(() => {
    // Initial animations
    setTimeout(() => {
      setShowLogo(false);
      setTimeout(() => setShowApp(true), 500);
    }, 2000);
  }, []);

  useEffect(() => {
    // Scroll to bottom when conversation updates
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const handleSubmit = async (message: string) => {
    setIsLoading(true);
    firstTaskId.current = null;
    isFirstTaskDone.current = false;
    const startTime = Date.now();
    
    // Create new conversation item with clean task state
    const newConversationItem: ConversationItem = {
      type: 'tasks',
      content: '',
      tasks: {},
      showLoading: true,
      hasFirstResponse: false
    };
    
    // Add user message and new conversation item
    setConversation(prev => [
      ...prev,
      { type: 'user', content: message },
      newConversationItem
    ]);

    try {
      // Get the previous response ID if any
      const lastTasksItem = [...conversation].reverse().find(item => item.type === 'tasks');
      const previousResponseId = lastTasksItem?.responseId;

      // Create streaming response
      const stream = await openai.responses.create({
        model: 'blast-default',
        input: message,
        stream: true,
        store: true,
        previous_response_id: previousResponseId
      });

      // Track tasks for this conversation item only
      const currentTasks: Record<string, TaskState> = {};
      let responseId: string | undefined;

      // Process each event from the stream
      for await (const event of stream as unknown as AsyncIterable<StreamEvent>) {
        // Get response ID from created event
        if (event.type === 'response.created' && event.response?.id) {
          responseId = event.response.id;
          // Update the response ID in conversation
          setConversation(prev => {
            const newConv = [...prev];
            const lastItem = newConv[newConv.length - 1];
            if (lastItem.type === 'tasks') {
              lastItem.responseId = responseId;
            }
            return newConv;
          });
        }

        if (event.type === 'response.output_text.delta') {
          const { item_id, delta } = event;
          const taskId = item_id?.split('_')[1];
          if (!taskId || !delta) continue;

          // Set first task ID if not set
          if (!firstTaskId.current) {
            firstTaskId.current = taskId;
          }

          // Initialize task if needed
          if (!currentTasks[taskId]) {
            currentTasks[taskId] = {
              taskId,
              updates: []
            };
          }

          // Add update if not duplicate
          const update: TaskUpdate = {
            content: delta,
            timestamp: Date.now(),
            type: delta.includes(' ') ? 'thought' : 'screenshot'
          };
          currentTasks[taskId].updates.push(update);

          // Show first response and hide loading when we get a screenshot
          if (update.type === 'screenshot') {
            setConversation(prev => {
              const newConv = [...prev];
              const lastItem = newConv[newConv.length - 1];
              if (lastItem.type === 'tasks') {
                lastItem.showLoading = false;
                lastItem.hasFirstResponse = true;
              }
              return newConv;
            });
          }

          // Update tasks in conversation
          setConversation(prev => {
            const newConv = [...prev];
            const lastItem = newConv[newConv.length - 1];
            if (lastItem.type === 'tasks') {
              // Create new tasks object to avoid state mixing
              lastItem.tasks = Object.fromEntries(
                Object.entries(currentTasks).map(([id, task]) => [
                  id,
                  {
                    taskId: task.taskId,
                    updates: [...task.updates],
                    finalResult: task.finalResult
                  }
                ])
              );
            }
            return newConv;
          });
        } else if (event.type === 'response.output_item.done' && event.item) {
          const taskId = event.item.id.split('_')[1];
          if (!taskId) continue;

          // Handle task completion
          if (firstTaskId.current === taskId) {
            isFirstTaskDone.current = true;
            setIsLoading(false);
          }

          // Get the final result from the task's updates
          const task = currentTasks[taskId];
          if (task) {
            // Get final result from the event's content
            const finalResult = event.item.content?.[event.item.content.length - 1]?.text;

            if (finalResult) {
              // Update the conversation with the final result
              setConversation(prev => {
                const newConv = [...prev];
                const lastItem = newConv[newConv.length - 1];
                if (lastItem.type === 'tasks') {
                  // For first task, set the conversation's final result and hide loading
                  if (firstTaskId.current === taskId) {
                    lastItem.finalResult = finalResult;
                    lastItem.showLoading = false;
                    lastItem.hasFirstResponse = true;
                  }
                  // Always update the task's final result
                  task.finalResult = finalResult;
                  lastItem.tasks = { ...currentTasks };
                }
                return newConv;
              });
            }
          }
        }
      }
      const endTime = Date.now();
      console.log(`Request completed in ${(endTime - startTime) / 1000} seconds at time (PST): ${new Date().toLocaleString('en-US', { timeZone: 'America/Los_Angeles' })}`);
    } catch (error) {
      console.error('Error:', error);
      setIsLoading(false);
      // Remove loading state on error
      setConversation(prev => {
        const newConv = [...prev];
        const lastItem = newConv[newConv.length - 1];
        if (lastItem.type === 'tasks') {
          lastItem.showLoading = false;
        }
        return newConv;
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#1f1f1f]">
      <AnimatePresence>
        {showLogo && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 flex items-center justify-center"
          >
            <Image
              src="/assets/blast_icon_only.svg"
              alt="BLAST Logo"
              width={120}
              height={120}
              className="w-32 h-32"
            />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showApp && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="min-h-screen"
          >
            <Header />
            
            <main className="pt-20 pb-32 px-6">
              <div className="max-w-4xl mx-auto">
                {conversation.map((item, index) => (
                  <div key={index} className="mb-8">
                    {item.type === 'user' ? (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-end"
                      >
                        <div className="bg-[#ffe067] text-black rounded-2xl px-6 py-3 max-w-[80%]">
                          {item.content}
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="space-y-6"
                      >
                        <AnimatePresence mode="wait">
                          {item.showLoading ? (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              transition={{ duration: 0.2 }}
                              className="ml-2"
                            >
                              <LoadingLogo onComplete={() => {
                                // Update the conversation item to remove loading logo from DOM
                                setConversation(prev => {
                                  const newConv = [...prev];
                                  const lastItem = newConv[newConv.length - 1];
                                  if (lastItem.type === 'tasks') {
                                    lastItem.showLoading = false;
                                  }
                                  return newConv;
                                });
                              }} />
                            </motion.div>
                          ) : item.hasFirstResponse && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="space-y-6"
                            >
                              {item.tasks && Object.values(item.tasks).some(task => hasScreenshots(task)) && (
                                <div className="grid grid-cols-3 gap-4">
                                  {Object.values(item.tasks)
                                    .filter(task => hasScreenshots(task))
                                    .map((task) => (
                                      <TaskBox
                                        key={task.taskId}
                                        taskId={task.taskId}
                                        updates={task.updates}
                                        finalResult={task.finalResult}
                                        mainTaskFinalResult={item.finalResult}
                                      />
                                    ))}
                                </div>
                              )}
                              {item.finalResult && (
                                <motion.div
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  className="text-gray-300 mt-4 whitespace-pre-wrap"
                                  transition={{ duration: 0.1 }}
                                >
                                  {item.finalResult}
                                </motion.div>
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    )}
                  </div>
                ))}
                <div ref={conversationEndRef} />
              </div>
            </main>

            <ChatInput onSubmit={handleSubmit} disabled={isLoading} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}