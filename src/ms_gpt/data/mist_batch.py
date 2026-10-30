"""Map MS-GPT spectrum tensors to the upstream MIST batch fields."""

def as_mist_batch(batch):
    num_peaks = batch["batch_peak_num"]
    return {
        "form_vec": batch["batch_peak_formulae_vector"],
        "types": batch["batch_peak_type"],
        "num_peaks": num_peaks,
        "intens": batch["batch_peak_intensity"],
        "ion_vec": batch["batch_peak_adduct"],
        "instruments": batch["batch_spec_instrument"],
    }
