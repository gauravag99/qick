"""Streaming acquisition mixin.

This module defines a new mixin class `StreamingAcquireMixin` which extends the
existing `AcquireMixin` (defined in `qick_asm.py`) to provide generator-based
streaming of raw readout data as it is polled from the hardware.

Key differences vs `AcquireMixin.acquire()`:

1. Data is yielded incrementally (per poll of new shots) without waiting for the
   entire round or software averaging to complete.
2. The streamed data is minimally processed: no averaging over reps, no length
   normalization, and (by default) no offset removal or thresholding.
3. Each yield packages the newly received points into shaped arrays of IQ data
   with dimensions `(new_points, reads_per_shot[ch], 2)` for each readout
   channel. Pre-shaped full-round buffers (filled with NaNs) are allocated at
   the start; newly arrived data replaces NaNs in-place.

Typical usage:

	prog = YourProgramSubclass(soccfg, ...)  # must include Acquire + setup_acquire
	prog.setup_acquire(counter_addr=..., loop_dims=[N], avg_level=0)
	for update in prog.stream_acquire(soc, rounds=R):
		if update['event'] == 'data':
			# per-poll incremental data
			partial = update['partial']  # dict: ch -> ndarray(new_points, nreads, 2)
			# optional full buffers so far:
			full = update.get('buffers')
		elif update['event'] == 'round-complete':
			round_raw = update['round_raw']  # list of full raw buffers for this round
		elif update['event'] == 'complete':
			all_rounds = update['rounds_raw']  # list of rounds' raw buffers
			break

Notes on shapes:
	* loop_dims: list of ints, flattened shot index progresses through the
	  cartesian product of these dimensions in row-major order.
	* For each readout channel i with reads_per_shot[i] triggers per shot, the
	  raw buffer shape matches AcquireMixin: (*loop_dims, reads_per_shot[i], 2).
	  We reshape internally to (total_shots * reads_per_shot[i], 2) for assignment.
	* Incremental payloads reshape the newly received flat slice to
	  (new_points, reads_per_shot[i], 2) before yielding.

Performance & memory:
	Preallocating NaN buffers avoids repeated concatenations and keeps constant
	memory. If copying large buffers every poll is undesired, set
	include_full=False to skip attaching them to each incremental yield.

Limitations:
	* This is designed for the 'accumulated' acquisition path; decimated and
	  'run_rounds' modes are not streamed here.
	* Requires firmware interface methods on `soc` matching those used by
	  AcquireMixin (`start_readout`, `poll_data`, etc.).
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Generator, Dict, Any, List

from qick import obtain  # for remote (Pyro) compatibility, mirrors AcquireMixin
from .qick_asm import AcquireMixin  # reuse existing logic & helpers

logger = logging.getLogger(__name__)


class StreamingAcquireMixin(AcquireMixin):
	"""Provides a generator interface for streaming raw readout data.

	Public generator method:
		stream_acquire(...): yields dictionaries describing acquisition events.

	Yielded event dicts can have the following forms:
		{'event': 'data', 'round': r, 'rep_slice': (start, stop),
		 'partial': {ch: ndarray(new_points, nreads, 2)},
		 'buffers': {ch: ndarray(*loop_dims, nreads, 2)} (optional)}

		{'event': 'round-complete', 'round': r, 'round_raw': [buffer_per_ch]}

		{'event': 'complete', 'rounds_raw': list_of_round_buffers}

	Parameters for stream_acquire mirror `acquire` with reduced processing:
		rounds: number of soft-average rounds to run (still no averaging in
				this method; we just separate rounds logically).
		remove_offset: if True, subtract expected IQ offset per channel when
				yielding incremental data (still no averaging).
		include_full: if True, include current full buffers (with NaNs for
				not-yet-arrived data) in each 'data' event yield.
		start_src: 'internal' or 'external' start mode forwarded to firmware.
	"""

	def stream_acquire(
		self,
		soc,
		rounds: int = 1,
		load_envelopes: bool = True,
		start_src: str = "internal",
		progress: bool = True,
		remove_offset: bool = True,
		len_normalize: bool = True,
		include_full: bool = False,
		return_end_of_exp_raw: bool = False,
	) -> Generator[Dict[str, Any], None, None]:
		"""Stream raw accumulated readout data.

		Yields dict events until completion. Final return (StopIteration value)
		is None; the final aggregated raw round buffers are provided in the
		last yield with event='complete'.
		"""
		# Validate required acquisition parameters were set via setup_acquire.
		if any([x is None for x in [self.counter_addr, self.loop_dims, self.avg_level]]):
			raise RuntimeError(
				"data dimensions need to be defined with setup_acquire() before calling stream_acquire()"
			)

		# Configure acquisition params similar to AcquireMixin.acquire but mark a distinct type.
		self.acquire_params = {
			'type': 'accumulated_stream',
			'soc': soc,
			'start_src': start_src,
			'rounds_remaining': rounds,
			'remove_offset': remove_offset,
			'hidereps': True,  # we manage optional progress display separately
		}

		total_count = int(np.prod(self.loop_dims))  # total flattened shots
		reads_per_shot = [ro['trigs'] for ro in self.ro_chs.values()]

		# Preallocate NaN-filled full buffers per channel (same shape as acc_buf in AcquireMixin)
		#TODO: consider keeping buffers for all rounds instead of just one?
		self.acc_buf = [
			np.full((*self.loop_dims, nreads, 2), np.nan, dtype=float)
			for nreads in reads_per_shot
		]

		# For completeness across rounds
		all_rounds_raw: List[List[np.ndarray]] = []

		# Load program & configure (do not load data memory yet; mirror acquire())
		self.config_all(soc, load_envelopes=load_envelopes, load_mem=False)

		# Progress bar for rounds only (optional)
		from tqdm.auto import tqdm  # local import; already used elsewhere
		self.round_track = tqdm(total=rounds, disable=not progress, desc="Rounds")

		for round_index in range(rounds):
			self.rounds_pbar = tqdm(total=total_count, disable=not progress, desc="Shots per Round", leave=False if rounds > 1 else True)
			# Prepare firmware for this round (subset of prepare_round logic for accumulated type)
			# Raw buffers already allocated; zero tProc counter & configure capture.
			self.config_bufs(soc, enable_avg=True, enable_buf=False)
			soc.reload_mem()
			soc.clear_tproc_counter(addr=self.counter_addr)
			soc.start_src(start_src)

			# Tracking variables
			count = 0  # number of shots (flattened) fully received

			# Start combined readout capture (similar to AcquireMixin.finish_round accumulated section)
			soc.start_readout(
				total_count,
				counter_addr=self.counter_addr,
				ch_list=list(self.ro_chs),
				reads_per_shot=reads_per_shot,
			)

			# Poll until all shots acquired
			while count < total_count:
				new_data = obtain(soc.poll_data())  # expect iterable of (new_points, (d, stats))
				new_data = list(new_data)  # materialize if generator
				# logger.info(f"new_data raw polled : {new_data} \n ================ \n ==============\n")
				if not new_data:
					continue  # nothing arrived yet
				for new_points, (d_list, stats) in new_data:
					if new_points <= 0:
						continue
					# Package incremental partial arrays per channel
					partial: Dict[int, np.ndarray] = {}
					for ii, nreads in enumerate(reads_per_shot):
						flat = d_list[ii]
						# logger.info(f"Channel {ii} polled flat data: {flat}")
						# Expect flat shape (new_points * nreads, 2)
						if flat.shape[0] != new_points * nreads:
							logger.error(
								"data size mismatch: new_points=%d, nreads=%d, data shape %s",
								new_points,
								nreads,
								flat.shape,
							)
						if count + new_points > total_count:
							logger.error(
								"got too much data: count=%d, new_points=%d, total_count=%d",
								count,
								new_points,
								total_count,
							)
						start_flat = count * nreads
						stop_flat = (count + new_points) * nreads
						# Assign into preallocated buffer (reshape view for contiguous assignment)
						self.acc_buf[ii].reshape((-1, 2))[start_flat:stop_flat] = flat
						# Reshape partial to (new_points, nreads, 2)
						partial_channel = flat.reshape(new_points, nreads, 2).astype(float)
						# logger.info(f"Channel {ii} partial reshaped data to shape {partial_channel.shape}: {partial_channel}")
						if len_normalize:
							# length normalization: divide by readout length
							ch_num = list(self.ro_chs.keys())[ii]
							length = self.ro_chs[ch_num]['length']
							# logger.info(f"Length normalization: readout length for ch {ch_num} is {length}")
							partial_channel /= length
							if remove_offset and not np.isnan(partial_channel).all():
								# subtract offset scaled to raw accumulated counts
								rocfg = self.ro_chs[ch_num].get('ro_config')
								# _ro_offset returns the DC offset
								offset = self._ro_offset(ch_num, rocfg)
								partial_channel -= offset
						partial[list(self.ro_chs.keys())[ii]] = partial_channel
					rep_slice = (count, count + new_points)
					count += new_points

					yield_event = {
						'event': 'data',
						'round': round_index,
						'rep_slice': rep_slice,
						'partial': partial,
					}
					if include_full:
						# Provide current state of full buffers (copy to avoid external mutation)
						yield_event['buffers'] = {
							ch: buf.copy() for ch, buf in zip(self.ro_chs.keys(), self.acc_buf)
						}
					self.rounds_pbar.update(new_points)
					yield yield_event

			# Round complete; snapshot raw buffers for this round (copy to freeze state)
			round_raw = [buf.copy() for buf in self.acc_buf]
			all_rounds_raw.append(round_raw)
			yield {
				'event': 'round-complete',
				'round': round_index,
				'round_raw': round_raw,
			}
			# Prepare for next round (reset buffers to NaN if more rounds remaining)
			if round_index < rounds - 1:
				for ii, nreads in enumerate(reads_per_shot):
					self.acc_buf[ii].fill(np.nan)

			self.rounds_pbar.close()
			self.round_track.update(1)
		self.round_track.close()
		# Final event summarizing all rounds' raw data
		if return_end_of_exp_raw:
			yield {
				'event': 'complete',
				'rounds_raw': all_rounds_raw,
			}


__all__ = ["StreamingAcquireMixin"]
