//! Errors from spectrum validation and diagnostics.

use core::fmt;

/// An invalid spectrum, policy, or diagnostic query.
#[derive(Clone, Debug, PartialEq)]
pub enum ChaosError {
    /// A diagnostic requires more levels than were supplied.
    TooFewLevels {
        /// Minimum number of levels.
        required: usize,
        /// Supplied number of levels.
        actual: usize,
    },
    /// An energy level is NaN or infinite.
    NonFiniteLevel {
        /// Index of the invalid level.
        index: usize,
    },
    /// A sequence claimed to be sorted decreases at one index.
    LevelsNotSorted {
        /// Index of the level smaller than its predecessor.
        index: usize,
    },
    /// A gap is not larger than the selected minimum.
    GapTooSmall {
        /// Index of the left level forming the gap.
        index: usize,
        /// Observed gap.
        gap: f64,
        /// Required strict lower bound.
        minimum: f64,
    },
    /// Subtracting adjacent finite levels overflowed.
    GapOverflow {
        /// Index of the left level forming the gap.
        index: usize,
    },
    /// A minimum-gap tolerance is negative or non-finite.
    InvalidMinimumGap,
    /// A local unfolding half-window must be at least one.
    InvalidLocalWindow,
    /// A form-factor time is NaN or infinite.
    NonFiniteTime {
        /// Index of the invalid time.
        index: usize,
    },
    /// Form-factor phase construction overflowed.
    PhaseOverflow {
        /// Time index.
        time_index: usize,
        /// Level index.
        level_index: usize,
    },
    /// A number-variance interval length is not finite and positive.
    InvalidIntervalLength,
    /// At least one number-variance window origin is required.
    EmptyOrigins,
    /// A number-variance origin is non-finite or its window crosses an edge.
    InvalidOrigin {
        /// Origin index.
        index: usize,
    },
}

impl fmt::Display for ChaosError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewLevels { required, actual } => write!(
                formatter,
                "diagnostic requires at least {required} levels, received {actual}"
            ),
            Self::NonFiniteLevel { index } => {
                write!(formatter, "energy level {index} is not finite")
            }
            Self::LevelsNotSorted { index } => write!(
                formatter,
                "energy levels decrease between indices {} and {index}",
                index - 1
            ),
            Self::GapTooSmall {
                index,
                gap,
                minimum,
            } => write!(
                formatter,
                "gap {index} is {gap:e}, but must be greater than {minimum:e}"
            ),
            Self::GapOverflow { index } => {
                write!(formatter, "gap {index} overflowed floating-point range")
            }
            Self::InvalidMinimumGap => {
                write!(formatter, "minimum gap must be finite and nonnegative")
            }
            Self::InvalidLocalWindow => {
                write!(
                    formatter,
                    "local unfolding half-window must be at least one"
                )
            }
            Self::NonFiniteTime { index } => {
                write!(formatter, "form-factor time {index} is not finite")
            }
            Self::PhaseOverflow {
                time_index,
                level_index,
            } => write!(
                formatter,
                "form-factor phase overflowed at time {time_index}, level {level_index}"
            ),
            Self::InvalidIntervalLength => {
                write!(formatter, "interval length must be finite and positive")
            }
            Self::EmptyOrigins => write!(formatter, "at least one window origin is required"),
            Self::InvalidOrigin { index } => write!(
                formatter,
                "window origin {index} is non-finite or crosses a spectrum edge"
            ),
        }
    }
}

impl std::error::Error for ChaosError {}
