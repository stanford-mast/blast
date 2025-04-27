import { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';

interface LoadingLogoProps {
  onComplete?: () => void;
}

export const LoadingLogo = ({ onComplete }: LoadingLogoProps) => {
  const [isVisible, setIsVisible] = useState(true);
  const [rotation, setRotation] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Start rotation animation interval
    intervalRef.current = setInterval(() => {
      setRotation(Math.random() * 360);
    }, 1500);

    // Set timeout to fade out and cleanup
    timeoutRef.current = setTimeout(() => {
      setIsVisible(false);
      if (onComplete) onComplete();
    }, 5000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [onComplete]);

  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ 
        opacity: 1,
        rotate: rotation
      }}
      transition={{ 
        opacity: { duration: 0.5 },
        rotate: { duration: 0.5, ease: "easeInOut" }
      }}
      className="flex items-center justify-center w-6 h-6"
    >
      <Image
        src="/assets/blast_icon_only.svg"
        alt="BLAST Logo"
        width={24}
        height={24}
        className="w-6 h-6"
      />
    </motion.div>
  );
};