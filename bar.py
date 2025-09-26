import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """An async progress bar that doesn't look terrible in your terminal.

    Shows progress with a visual bar, percentage complete, timing info, and
    any custom metrics you want to display. Built for async/await and handles
    cleanup automatically when done.

    Attributes:
        iterations (int): Total number of items to process.
        title (str): Display name for this progress bar.
        steps (int): Width of the progress bar in characters.
        items (Dict[str, str]): Additional metrics to display alongside progress.
    """

    def __init__(
            self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """Initialize a new progress bar.

        Args:
            iterations: Total count of items to process. Must be positive.
            title: Label for this operation, appears as "Training: [####...]".
            steps: Character width of the progress bar. Higher values give
                smoother visual updates but with slight performance cost.

        Raises:
            ValueError: When iterations is not a positive integer.
        """
        # Total items we need to process
        self.iterations: int = iterations

        # Display label for this operation
        self.title: str = title

        # Character width of the progress bar
        self.steps: int = steps

        # Dictionary for storing custom metrics to display
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """Refresh the progress bar with current progress information.

        Recalculates completion percentage, processing speed, estimated time
        remaining, and redraws the entire progress display.

        Args:
            batch: Number of items completed so far.
            time: Start timestamp (from time.time() or event loop timer).
            final: When True, adds newline after final update.

        Note:
            Uses asyncio event loop time for consistent timing calculations.
        """
        # Calculate how long this has been running
        elapsed: float = np.subtract(
            asyncio.get_event_loop().time(), time
        )

        # Calculate completion percentage
        percentage: float = np.divide(batch, self.iterations)

        # Calculate processing rate (items per second)
        throughput: np.array = np.where(
            np.greater(elapsed, 0),
            np.floor_divide(batch, elapsed),
            0
        )

        # Estimate remaining time based on current progress
        eta: np.array = np.where(
            np.greater(batch, 0),
            np.divide(
                np.multiply(elapsed, np.subtract(self.iterations, batch)),
                batch
            ),
            0,
        )

        # Construct the visual progress bar
        bar: str = chars.add(
            "|",
            chars.add(
                # Fill completed portion with hash marks
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Fill remaining portion with spaces
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps,
                                np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Add progress counter like 042/100
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Build complete output line
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Core progress info: title, bar, percentage, timing
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",

                        # Append custom metrics if any exist
                        np.where(
                            np.greater(np.size(self.items), 0),
                            chars.add(
                                " (",
                                chars.add(
                                    # Format custom metrics as "key: value, key: value"
                                    ", ".join(
                                        [
                                            f"{name}: {value}"
                                            for name, value in self.items.items()
                                        ]
                                    ),
                                    ")",
                                ),
                            ),
                            "",
                        ),
                    ),
                    "",
                ),
                "",
            )
        )

        # Add newline when completely finished
        if final:
            sys.stdout.write("\n")

        # Force output to appear immediately
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """Update custom metrics displayed alongside the progress bar.

        Useful for showing dynamic values like loss, accuracy, learning rate,
        or other relevant statistics during processing.

        Args:
            **kwargs: Metrics to display as key=value pairs. Example:
                postfix(loss=0.42, accuracy=0.89, lr=0.001)
        """
        # Update our metrics dictionary with new values
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """Enable usage with async context managers.

        Returns:
            The Bar instance for use within the async with block.

        Example:
            async with Bar(100, "Processing") as pbar:
                for i in range(100):
                    await pbar.update(i+1, start_time)
        """
        return self

    async def __aexit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """Handle cleanup when exiting the async context manager.

        On normal completion, displays final 100% progress update.
        On exceptions, shows a warning about the error that occurred.

        Args:
            exc_type: Exception class if an error occurred, None otherwise.
            exc_val: The exception instance that was raised.
            exc_tb: Traceback object containing error details.
        """
        if exc_type is None:
            # Normal completion - show final progress update
            await self.update(
                self.iterations,
                asyncio.get_event_loop().time(),
                final=True
            )
        else:
            # Exception occurred - warn user about the error
            warnings.warn(
                f"\n{self.title} encountered an error: {exc_val}"
            )