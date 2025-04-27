import { useState, useRef, useEffect } from 'react';
import { ArrowUpCircleIcon } from '@heroicons/react/24/solid';

interface ChatInputProps {
  onSubmit: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput = ({ onSubmit, disabled }: ChatInputProps) => {
  const [message, setMessage] = useState('');
  const [buttonProgress, setButtonProgress] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const inputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Calculate progress based on message length (0 to 20 characters)
    const length = message.trim().length;
    const progress = Math.min(length / 20, 1);
    setButtonProgress(progress);
  }, [message]);

  const handleSubmit = () => {
    if (message.trim() && !disabled) {
      onSubmit(message.trim());
      setMessage('');
      if (inputRef.current) {
        inputRef.current.textContent = '';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const text = e.clipboardData.getData('text');
    document.execCommand('insertText', false, text);
  };

  // Calculate button color based on progress
  const buttonColor = buttonProgress === 0 
    ? 'text-gray-500' 
    : `text-[#ffe067]`;

  const isInputDisabled = disabled || !message.trim();

  // Calculate input background color based on state
  const getInputBgColor = () => {
    if (disabled) return 'bg-[#262626]';
    if (isFocused) return 'bg-[#2a2a2a]';
    if (isHovered) return 'bg-[#272727]';
    return 'bg-[#262626]';
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-[#1f1f1f] p-8">
      <div className="max-w-4xl mx-auto flex items-center gap-4">
        <div className="relative flex-1">
          <div
            ref={inputRef}
            contentEditable={!disabled}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            onInput={(e) => setMessage(e.currentTarget.textContent || '')}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`chat-input min-h-[56px] max-h-[200px] overflow-y-auto whitespace-pre-wrap break-words rounded-3xl px-6 py-4 text-[15px] transition-colors duration-150 ${
              getInputBgColor()
            } ${
              disabled ? 'cursor-not-allowed opacity-50' : 'cursor-text'
            }`}
            style={{
              outline: 'none',
            }}
          />
          {!message && (
            <div className={`absolute left-6 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none text-[15px] ${
              disabled ? 'opacity-50' : ''
            }`}>
              Do anything
            </div>
          )}
        </div>
        <button
          onClick={handleSubmit}
          disabled={isInputDisabled}
          className={`submit-button flex items-center justify-center transition-all duration-300 ${buttonColor} ${
            isInputDisabled ? 'cursor-not-allowed' : 'cursor-pointer'
          } hover:opacity-90`}
          style={{
            opacity: isInputDisabled ? 0.3 : (0.4 + (buttonProgress * 0.6)), // Lower opacity when disabled
          }}
        >
          <ArrowUpCircleIcon className="w-[52px] h-[52px]" />
        </button>
      </div>
    </div>
  );
};