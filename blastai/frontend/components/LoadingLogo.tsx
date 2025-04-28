import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';

interface LoadingLogoProps {
  onComplete?: () => void;
}

export const LoadingLogo = ({ onComplete }: LoadingLogoProps) => {
  const [rotation, setRotation] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Start rotation animation interval
    intervalRef.current = setInterval(() => {
      setRotation(Math.random() * 360);
    }, 1500);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{
        opacity: 1,
        rotate: rotation
      }}
      exit={{
        opacity: 0,
        transition: { duration: 0.2 }
      }}
      transition={{
        opacity: { duration: 0.2 },
        rotate: { duration: 0.5, ease: "easeInOut" }
      }}
      onAnimationComplete={(definition) => {
        if (definition === "exit" && onComplete) {
          onComplete();
        }
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