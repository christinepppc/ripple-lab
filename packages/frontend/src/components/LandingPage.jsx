import React, { useEffect, useState } from 'react';
import './LandingPage.css';

const LandingPage = ({ onEnter }) => {
  const [hasPassedBoundary, setHasPassedBoundary] = useState(false);

  useEffect(() => {
    const landingHeight = window.innerHeight;

    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      
      // If we've scrolled past the landing page boundary (50% of viewport)
      if (currentScrollY >= landingHeight * 0.5) {
        if (!hasPassedBoundary) {
          setHasPassedBoundary(true);
          // Lock scroll position at boundary and notify parent
          window.scrollTo({
            top: landingHeight,
            behavior: 'smooth'
          });
          // Notify parent component to show app
          if (onEnter) {
            setTimeout(() => onEnter(), 300); // Wait for scroll to complete
          }
        }
      }
    };

    // Prevent scrolling back up past boundary
    const preventScrollBack = (e) => {
      if (hasPassedBoundary) {
        const currentScrollY = window.scrollY;
        if (currentScrollY < landingHeight) {
          e.preventDefault();
          e.stopPropagation();
          window.scrollTo(0, landingHeight);
          return false;
        }
      }
    };

    // Handle scroll events
    const scrollHandler = () => {
      handleScroll();
      if (hasPassedBoundary && window.scrollY < landingHeight) {
        window.scrollTo(0, landingHeight);
      }
    };

    window.addEventListener('scroll', scrollHandler, { passive: false });
    window.addEventListener('wheel', preventScrollBack, { passive: false });
    window.addEventListener('touchmove', preventScrollBack, { passive: false });
    
    // Prevent keyboard scrolling back
    const keyHandler = (e) => {
      if (hasPassedBoundary && (e.key === 'ArrowUp' || e.key === 'PageUp' || e.key === 'Home')) {
        if (window.scrollY < landingHeight) {
          e.preventDefault();
          window.scrollTo(0, landingHeight);
        }
      }
    };
    window.addEventListener('keydown', keyHandler);
    
    return () => {
      window.removeEventListener('scroll', scrollHandler);
      window.removeEventListener('wheel', preventScrollBack);
      window.removeEventListener('touchmove', preventScrollBack);
      window.removeEventListener('keydown', keyHandler);
    };
  }, [hasPassedBoundary]);

  // Hide landing page once past boundary
  if (hasPassedBoundary) {
    return null;
  }

  return (
    <div className="landing-container">
      <div className="landing-hero">
        <div className="landing-content">
          <img 
            src="/logo.png" 
            alt="Ripple Lab Logo" 
            className="landing-logo"
            onError={(e) => {
              // Fallback if logo doesn't load
              e.target.style.display = 'none';
            }}
          />
          <h1 className="landing-title">Ripple Lab</h1>
          <p className="landing-subtitle">Neural Ripple Analysis Platform</p>
          <div className="scroll-indicator">
            <span>Scroll to begin</span>
            <div className="scroll-arrow">↓</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;

