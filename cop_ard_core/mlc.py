
import numpy as np
import itertools
import xarray as xr
from s1_ard_core.computing.computation import map_blocks
from s1_ard_core.computing.logger import logger, timeit
from s1_ard_core.computing.io import read,write,normpath,save_empty_dataset,get_storage_options
import cop_ard_core.utils
from dask.distributed import print


def single_cmlc(burst_id:str,cstack:xr.DataArray,multilook:dict,working_directory:str,revisit_cycle:np.timedelta64,memory_param:float = 0.95,coherence_threshold:float=0.1,stable_reference_step:int = 10,max_past_stable_history:int=10, last_reference:xr.DataArray | None = None, stable_reference: xr.DataArray | None = None,**kwargs):
    path = f"bursts/{burst_id}" 
    data_path = normpath(f"{working_directory}/{path}")
    if set(multilook) != {"azimuth_time", "slant_range_time"}:
        raise ValueError(f"Wrong keys: {set(multilook)}. Expected azimuth_time, slant_range_time")
    steps = {k:cstack.chunksizes[k][0] // v * v for k,v in multilook.items()}
    slices = {k:slice(0,cstack[k].size // v * v) for k,v in multilook.items()}
    # slices = {k:slice(0,2040) for k,v in multilook.items()}
    chunks = {k:v[0] for k,v in cstack.chunksizes.items()}
    dims = cstack.dims
    cstack_sel = cstack.isel(slices).drop_vars([k for k in cstack.coords if k not in dims]).sortby("acquisition_date")
    
    # Calculate how much to add
    pad = {k: (0,cstack[k].size - cstack_sel[k].size) for k in multilook.keys()}
    acquisition_dates = cstack_sel.acquisition_date.values
    stable_reference_date = np.min(acquisition_dates.astype("datetime64[ns]"))
    if last_reference is not None:
        last_reference = last_reference.isel(slices).drop_vars([k for k in last_reference.coords if k not in dims])
    if stable_reference is not None:
        stable_reference = stable_reference.isel(slices).drop_vars([k for k in stable_reference.coords if k not in dims])
        stable_reference_date = np.min(stable_reference.acquisition_date.astype("datetime64[ns]"))
    
    elapsed_cycles = (acquisition_dates.astype("datetime64[ns]") - stable_reference_date) / revisit_cycle.astype("timedelta64[ns]")
    stable_reference_dates = []
    threshold = stable_reference_step
    for i,cycle in enumerate(elapsed_cycles):
        if cycle > threshold:
            stable_reference_dates.append(acquisition_dates[i])
            threshold += stable_reference_step
    stable_reference_dates = np.array(stable_reference_dates)

    acquisition_date_offset = 0
    if stable_reference is None:
        acquisition_date_offset = np.searchsorted(acquisition_dates, stable_reference_dates[0])

    data_vars = ["cstack"]
    coords = {k:v for k,v in cstack.drop_vars([k for k in cstack.coords if k not in dims]).coords.items()}
    coords["acquisition_date"] = stable_reference_dates
    shape = [coords[i].size for i in ["polarization","acquisition_date","azimuth_time","slant_range_time"]]
    save_empty_dataset(
        data_vars=data_vars,
        dims={k: dims for k in data_vars},
        coords=coords,
        shape={k: shape for k in data_vars},
        chunks=chunks,
        dtype={k: "complex64" for k in data_vars},
        skip_read=True,
        data_path=f"{data_path}/stable_reference",
        storage_options=get_storage_options(kwargs),
    )

    current_reference,start_date = get_current_reference(cstack_sel, last_reference,stable_reference)
    y,x = [np.arange(0,current_reference[k].size // v).repeat(v) for k,v in multilook.items()]    
    
    data_vars = ["phase","coherence"]
    coords = {k:current_reference[k].coarsen({k:v}).mean().values for k,v in multilook.items()}
    coords["polarization"] = current_reference.polarization.values

    for offset,file_name in zip([0,acquisition_date_offset],["short_term_cmlc","long_term_cmlc"]):
        coords["acquisition_date"] = cstack_sel.acquisition_date.values[offset:]
        shape = [coords[i].size for i in ["polarization","acquisition_date","azimuth_time","slant_range_time"]]
        save_empty_dataset(
            data_vars=data_vars,
            dims={k: dims for k in data_vars},
            coords=coords,
            shape={k: shape for k in data_vars},
            chunks=chunks,
            dtype={k: "float32" for k in data_vars},
            skip_read=True,
            data_path=f"{data_path}/{file_name}",
            storage_options=get_storage_options(kwargs),
        )

    chunks.pop("acquisition_date")
    stable_cstack = stable_reference 
    for date_idx,date in enumerate(cstack_sel.acquisition_date.values):
        print("="*50)
        print(date)
        if date == start_date:
            print("Skip")
            continue
        
        slc = cstack_sel.sel(acquisition_date = date)
        normalized_interf = cop_ard_core.utils.get_normalized_interf(slc,current_reference,multilook).compute()
        coherence_interf = np.abs(normalized_interf)
        phase_interf = xr.apply_ufunc(np.angle, normalized_interf, dask="parallelized")
        ds = xr.Dataset({"coherence":coherence_interf,"phase":phase_interf}).expand_dims({"acquisition_date":1}).transpose(*dims)
        write(ds,f"{data_path}/short_term_cmlc",mode="r+",region = {"acquisition_date":slice(date_idx, date_idx + 1)},skip_read=True,consolidated=False)

        phasor_interf = normalized_interf / coherence_interf
        phasor_interf = phasor_interf.isel(azimuth_time = y,slant_range_time = x).drop_vars(phasor_interf.coords).chunk(chunks)
        correction = slc * phasor_interf.conj()
        current_reference = memory_param*current_reference + correction
        
        if date in stable_reference_dates:
            print(f"Updating stable reference : {date}")
            idx = np.searchsorted(stable_reference_dates, date)
            stable_reference = current_reference.pad(pad,constant_values=np.nan).chunk(chunks).expand_dims({"acquisition_date":1}).transpose(*dims).to_dataset(name="cstack")
            write(stable_reference,f"{data_path}/stable_reference",mode="r+",region = {"acquisition_date":slice(idx, idx + 1)},skip_read=True,consolidated=False)
            stable_cstack = read(f"{data_path}/stable_reference",consolidate_metadata=False).cstack.isel({**slices,"acquisition_date":slice(0,idx+1)})
            if last_reference is not None:
                stable_cstack = xr.concat([last_reference,stable_cstack],dim="acquisition_date")
            
        if stable_cstack is not None:
            stable_cstack = stable_cstack.sortby("acquisition_date").isel(acquisition_date=slice(-max_past_stable_history,None))

            normalized_interf = cop_ard_core.utils.get_normalized_interf(stable_cstack.isel(acquisition_date=0),current_reference,multilook).compute()
            coherence_interf = np.abs(normalized_interf)
            phase_interf = xr.apply_ufunc(np.angle, normalized_interf, dask="parallelized")
            ds = xr.Dataset({"coherence":coherence_interf,"phase":phase_interf}).expand_dims({"acquisition_date":1}).transpose(*dims)
            write(ds,f"{data_path}/long_term_cmlc",mode="r+",region = {"acquisition_date":slice(date_idx -acquisition_date_offset, date_idx + 1 -acquisition_date_offset)},skip_read=True,consolidated=False)            



            drift_interf = xr.full_like(normalized_interf,np.nan)
            for iteration,stable_date in enumerate(stable_cstack.acquisition_date.values):
                print('*'*50)
                stable_reference = stable_cstack.sel(acquisition_date=stable_date)    
                normalized_interf = cop_ard_core.utils.get_normalized_interf(stable_reference,current_reference,multilook).values
                is_coherent = np.abs(normalized_interf) >= coherence_threshold
                is_empty = np.isnan(drift_interf)
                mask = is_coherent * is_empty
                drift_interf.data[mask] = normalized_interf[mask]
                if not np.isnan(correction).any():
                    print(f'Completed at Iteration {iteration}')
                    break

            

            drift_interf = drift_interf / np.abs(drift_interf)
            drift_interf = drift_interf.isel(azimuth_time = y,slant_range_time = x).drop_vars(drift_interf.coords).chunk(chunks)
            current_reference = current_reference * drift_interf.conj()
    
    return 1




def single_cmlc_old(burst_id:str,cstack:xr.DataArray,multilook:dict,working_directory:str,revisit_cycle:np.timedelta64,memory_param:float = 0.95,coherence_threshold:float=0.1,stable_reference_step:int = 10, last_reference:xr.DataArray | None = None, stable_reference: xr.DataArray | None = None):
    steps = {k:cstack.chunksizes[k][0] // v * v for k,v in multilook.items()}
    slices = {k:slice(0,cstack[k].size // v * v) for k,v in multilook.items()}
    cstack_sel = cstack.isel(slices)
    
    args = []
    obj = [cstack_sel]
    if last_reference is not None:
        last_reference = last_reference.isel(slices)
        obj += [last_reference]
    else:
        args += [None]

    if stable_reference is not None:
        stable_reference = stable_reference.isel(slices)
        obj += [stable_reference]
    else:
        args += [None]

    optimal_chunks = {}
    for k in cstack.dims:
        size = cstack_sel[k].size
        if k in multilook:
            a = [[s,s+steps[k] if s+steps[k]<size else size -1 ] for s in np.arange(0,size,steps[k])]
        else:
            a = [[0,size-1]]
        optimal_chunks[k] = a

    r = np.array(list(itertools.product(*[v for v in optimal_chunks.values()])))
    optimal_chunks = {k: r[:, en].tolist() for en, k in enumerate(optimal_chunks.keys())}

    print(cstack_sel)

    return cstack_sel




    args += [revisit_cycle,multilook,memory_param,coherence_threshold,stable_reference_step]
    results = map_blocks(compute_cmlc_blk,obj = obj,args = args,chunks=optimal_chunks,enable_dask=False)









def compute_cmlc_blk(cstack:xr.DataArray,last_reference:xr.DataArray | None, stable_reference:xr.DataArray | None, revisit_cycle:np.timedelta64,multilook:dict,memory_param:float,coherence_threshold:float,stable_reference_step:int,chunk = None):
    print("*"*50)
    print(chunk)

    current_reference = get_current_reference(cstack, last_reference,stable_reference)
    y,x = [np.arange(0,current_reference[k].size // v).repeat(v) for k,v in multilook.items()]    
    counter = 0

    for date in sorted(cstack.acquisition_date.values):
        print("="*50)
        print(date)
        if date == current_reference.acquisition_date.values:
            print("skip")
            continue
        slc = cstack.sel(acquisition_date = date).compute()
        normalized_interf = cop_ard_core.utils.get_normalized_interf(slc,current_reference,multilook)
        module_normalize_interf = np.abs(normalized_interf)
        short_term_coherence = module_normalize_interf
        phasor_interf = normalized_interf / module_normalize_interf
        phasor_interf = phasor_interf.isel(azimuth_time = y,slant_range_time = x).drop_vars(phasor_interf.coords)    
        correction = slc * phasor_interf.conj()
        current_reference = memory_param*current_reference + correction

        if counter % stable_reference_step == 0:
            print(f"Updating stable history : {counter}")
            stable_reference_history += [current_reference]
            correction = xr.ones_like(normalized_interf)
            for count,past_reference in enumerate(current_reference[::-1]):    
                normalized_interf = get_normalized_interf(past_reference,current_reference).compute().chunk(chunks)
                ind = np.where(np.abs(normalized_interf) >= coherence_threshold)
                correction[ind] = normalized_interf[ind]
            
                
                

        counter += 1

        if counter ==16:
            break

    
    

def get_current_reference(cstack:xr.DataArray, last_reference:xr.DataArray | None,stable_reference:xr.DataArray | None):
    date = np.min(cstack.acquisition_date.values.astype("datetime64"))
    date_l = np.max(last_reference.acquisition_date.values.astype("datetime64")) if last_reference is not None else None
    date_s = np.max(stable_reference.acquisition_date.values.astype("datetime64")) if stable_reference is not None else None
    current_reference = cstack.sel(acquisition_date=date.astype(str))

    if (date_l is not None) and (date_s is not None):
        if date_l > date_s:
            if date_l < date:
                current_reference = last_reference.sel(acquisition_date=date_l.astype(str))  
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        else:
            if date_s < date:
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))  
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")
    else:
        if date_l is not None:
            if date_l < date:
                current_reference = last_reference.sel(acquisition_date=date_l.astype(str))  
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        if date_s is not None:
            if date_s < date:
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))  
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")

    date = current_reference.acquisition_date.values
    current_reference = current_reference.drop_vars({k for k in current_reference.coords.keys() if k not in ["polarization","azimuth_time","slant_range_time"]})
    return current_reference,date