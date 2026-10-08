// The "o" of the page title, drawn as an eye that blinks now and then.
// The blink is in styles.css, and is off for people who ask for less motion.
export function TitleEye() {
  return (
    <svg className="title-eye" viewBox="0 0 100 70" aria-hidden="true">
      <path d="M6 35 Q50 -6 94 35 Q50 76 6 35 Z" fill="none" stroke="currentColor" strokeWidth="11"
            strokeLinejoin="round" />
      <circle className="title-eye-iris" cx="50" cy="35" r="17" />
      <circle className="title-eye-pupil" cx="50" cy="35" r="7" />
    </svg>
  );
}
