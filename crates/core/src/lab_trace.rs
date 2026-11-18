//! Experimental persistent Trellis samples. No GPU addresses cross the file boundary.
#![cfg(feature = "cuda")]
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use serde_json::{json, Value};
use crate::model_load::{Model, ffi::IgnisModel};
use crate::seq::{SeqPool, ffi::IgnisSeqPool};

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct GqaTensor { dtype:u32, shape:[i32;4] }
#[repr(C)]
struct GqaView { version:u32, layer:u32, max_keys:u32, scale:f32, tensors:[GqaTensor;15] }
#[repr(C)]
struct View {
    name:*const c_char, trellis_offset:u64, suh_offset:u64, svh_offset:u64,
    k2:u32, input:u32, output:u32, tile_stride:u32,
}
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Diagnostics {
    mismatches:u64, total_elements:u64, finite_elements:u64,
    reference_nan:u64, reference_inf:u64, actual_nan:u64, actual_inf:u64,
    max_abs_error:f64, squared_error:f64, squared_reference:f64,
}
impl Diagnostics {
    fn json(&self)->Value {
        let relative_l2=if self.finite_elements==0 {None}
            else if self.squared_reference>0.0 {Some((self.squared_error/self.squared_reference).sqrt())}
            else if self.squared_error==0.0 {Some(0.0)} else {None};
        json!({"mismatches":self.mismatches,"total_elements":self.total_elements,"finite_elements":self.finite_elements,
            "reference_nan":self.reference_nan,"reference_inf":self.reference_inf,
            "actual_nan":self.actual_nan,"actual_inf":self.actual_inf,
            "max_abs_error":if self.finite_elements>0 {Some(self.max_abs_error)} else {None},
            "relative_l2":relative_l2,"relative_l2_zero_reference":self.finite_elements>0 && self.squared_reference==0.0,
            "error_scope":"finite pairs across checked executions"})
    }
}
#[derive(Clone, Copy)]
#[repr(C)]
struct SegmentProfile {kind:u32,tokens:u32,batch:u32,hidden:u32,intermediate:u32,max_keys:u32,slots:u32,flags:u32,rms_eps:f32}
#[repr(C)]
struct SegmentView {name:*const c_char,flags:u32,dtype:u32,present:u32,shape:[i32;4]}

unsafe extern "C" {
    fn ignis_lab_capture_start_filtered(m:*mut IgnisModel,p:*mut IgnisSeqPool,dir:*const c_char,pin:u64,disk:u64,
        names:*const c_char,token_filter:*const u32,token_filter_count:u32)->i32;
    fn ignis_lab_capture_stop(m:*mut IgnisModel,p:*mut IgnisSeqPool)->i32;
    fn ignis_lab_trace_context(m:*mut IgnisModel,request:u32,phase:u32,position:u64)->i32;
    fn ignis_lab_trace_flush(m:*mut IgnisModel)->i32;
    fn ignis_lab_profile_status()->i32;
    fn ignis_lab_profile_push(label:*const c_char)->i32;
    fn ignis_lab_profile_pop();
    fn ignis_lab_profiler_start()->i32;
    fn ignis_lab_profiler_stop()->i32;
    fn ignis_lab_capture_minimum_filtered(m:*mut IgnisModel,names:*const c_char,tokens:u64,requests:u64,
        token_filter:*const u32,token_filter_count:u32,bytes:*mut u64)->i32;
    fn ignis_lab_trace_error()->*const c_char;
    fn ignis_lab_gqa_implementation(m:*mut IgnisModel)->*const c_char;
    fn ignis_lab_gqa_candidate_applied(m:*mut IgnisModel)->i32;
    fn ignis_lab_segment_implementation(m:*mut IgnisModel)->*const c_char;
    fn ignis_lab_segment_workspace_capacity(m:*mut IgnisModel,kind:u32)->u64;
    fn ignis_lab_segment_replay(m:*mut IgnisModel,layer:u32,p:*const SegmentProfile,views:*const SegmentView,count:u32,
        blob:*const c_void,bytes:u64,warmup:u32,samples:u32,check:u32,times:*mut f32,checked:*mut u32,
        diagnostics:*mut Diagnostics,per_binding:*mut Diagnostics,state_mismatches:*mut u64,state_bytes:*mut u64,
        binding_state_mismatches:*mut u64,binding_state_bytes:*mut u64)->i32;
    fn ignis_lab_gqa_replay(m:*mut IgnisModel,view:*const GqaView,blob:*const c_void,bytes:u64,
        warmup:u32,samples:u32,check:u32,times:*mut f32,checked:*mut u32,diagnostics:*mut Diagnostics,
        state_mismatches:*mut u64,state_bytes:*mut u64)->i32;
    fn ignis_lab_replay_diagnostics(m:*mut IgnisModel,w:*const View,up:*const View,t:u32,ldy:u32,epilogue:u32,
        x:*const c_void,initial:*const c_void,reference:*const c_void,warmup:u32,samples:u32,check:u32,
        times:*mut f32,mismatches:*mut u64,checked:*mut u32,diagnostics:*mut Diagnostics)->i32;
}
fn checked(rc:i32)->Result<(),String>{
    if rc==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(ignis_lab_trace_error())}.to_string_lossy().into_owned())}
}
pub fn start(model:&mut Model,pool:&mut SeqPool,path:&Path,pin:u64,disk:u64,names:&str)->Result<(),String>{
    start_filtered(model,pool,path,pin,disk,names,&[])
}
pub fn start_filtered(model:&mut Model,pool:&mut SeqPool,path:&Path,pin:u64,disk:u64,names:&str,token_filter:&[u32])->Result<(),String>{
    let count=u32::try_from(token_filter.len()).map_err(|e|e.to_string())?;
    let dir=CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e|e.to_string())?;
    let names=CString::new(names).map_err(|e|e.to_string())?;
    checked(unsafe{ignis_lab_capture_start_filtered(model.handle(),pool.handle(),dir.as_ptr(),pin,disk,names.as_ptr(),token_filter.as_ptr(),count)})
}
pub fn stop(model:&mut Model,pool:&mut SeqPool)->Result<(),String>{
    checked(unsafe{ignis_lab_capture_stop(model.handle(),pool.handle())})
}
pub fn context(model:&Model,request:u32,phase:u32,position:u64)->Result<(),String>{
    checked(unsafe{ignis_lab_trace_context(model.handle(),request,phase,position)})
}
pub fn flush(model:&Model)->Result<(),String>{checked(unsafe{ignis_lab_trace_flush(model.handle())})}
/// Semantic range labels are submission identities, never GPU durations.
/// The guard stays on its creating thread and closes on early return.
pub struct ProfileRange {active:bool,_thread:std::marker::PhantomData<*mut ()>}
impl Drop for ProfileRange {
    fn drop(&mut self) {if self.active {unsafe{ignis_lab_profile_pop()};}}
}
pub fn profile_range(label:&str)->Result<ProfileRange,String>{
    let status=unsafe{ignis_lab_profile_status()};
    if status<0 {return Err("NVTX3 headers unavailable at engine build; set NVTX3_INCLUDE_DIR and rebuild".into());}
    let active=if status==0 {false} else {
        let label=CString::new(label).map_err(|e|e.to_string())?;
        unsafe{ignis_lab_profile_push(label.as_ptr())!=0}
    };
    Ok(ProfileRange{active,_thread:std::marker::PhantomData})
}
pub fn profile_status()->i32 {unsafe{ignis_lab_profile_status()}}
/// Bound an external Nsight Systems capture around selected replay commands.
pub fn profiler_start()->Result<(),String>{checked(unsafe{ignis_lab_profiler_start()})}
pub fn profiler_stop()->Result<(),String>{checked(unsafe{ignis_lab_profiler_stop()})}
fn number(v:&Value,key:&str)->Result<u64,String>{v[key].as_u64().ok_or_else(||format!("invalid trace field {key}"))}
fn small(v:&Value,key:&str)->Result<u32,String>{u32::try_from(number(v,key)?).map_err(|e|e.to_string())}
fn view(v:&Value,name:&CString)->Result<View,String>{Ok(View{name:name.as_ptr(),
    trellis_offset:number(v,"trellis_offset")?,suh_offset:number(v,"suh_offset")?,svh_offset:number(v,"svh_offset")?,
    k2:small(v,"k2")?,input:small(v,"in")?,output:small(v,"out")?,tile_stride:small(v,"tile_stride")?})}

/// Regression callers check every execution, including warmups. Exploration reports
/// differences across every execution without truncating the timing sample set.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum ReplayCheck { FirstExecution = 0, EveryExecution = 1, Exploration = 2 }

pub fn replay(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32)->Result<Value,String>{
    replay_with_check(model,data,record,warmup,samples,ReplayCheck::EveryExecution)
}

pub fn replay_with_check(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32,check:ReplayCheck)->Result<Value,String>{
    if samples==0 || samples>10000 || warmup>10000 {return Err("invalid sample counts".into());}
    if record["kind"]=="segment" {return replay_segment(model,data,record,warmup,samples,check);}
    if record["kind"]=="gqa_attention" {return replay_gqa(model,data,record,warmup,samples,check);}
    let name=CString::new(record["weight"]["name"].as_str().ok_or("missing weight name")?).map_err(|e|e.to_string())?;
    let w=view(&record["weight"],&name)?;
    let upname=CString::new(record["up"]["name"].as_str().unwrap_or("")).map_err(|e|e.to_string())?;
    let up=if record["up"].is_null(){None}else{Some(view(&record["up"],&upname)?)};
    let t=small(record,"tokens")?;let ldy=small(record,"ldy")?;
    let nx=number(record,"x_bytes")?;let ny=number(record,"y_bytes")?;
    let bytes=nx.checked_add(ny.checked_mul(2).ok_or("sample size overflow")?).ok_or("sample size overflow")?;
    if t==0 || t>8192 || w.input==0 || w.input>524288 || w.output==0 || w.output>524288 ||
        nx!=2*u64::from(t)*u64::from(w.input) || ny!=2*u64::from(t)*u64::from(w.output) ||
        bytes!=number(record,"bytes")? || bytes>2u64.pow(31) {return Err("invalid sample geometry".into());}
    let offset=number(record,"offset")?;
    let mut file=std::fs::File::open(data).map_err(|e|e.to_string())?;
    if offset.checked_add(bytes).ok_or("offset overflow")?>file.metadata().map_err(|e|e.to_string())?.len(){return Err("truncated trace data".into());}
    file.seek(SeekFrom::Start(offset)).map_err(|e|e.to_string())?;
    let mut buf=vec![0u8;bytes as usize];file.read_exact(&mut buf).map_err(|e|e.to_string())?;
    let mut times=vec![0f32;samples as usize];let mut mismatches=0;let mut executions_checked=0;
    let mut diagnostics=Diagnostics::default();
    checked(unsafe{ignis_lab_replay_diagnostics(model.handle(),&w,up.as_ref().map_or(std::ptr::null(),|v|v),t,ldy,
        small(record,"epilogue")?,buf.as_ptr().cast(),buf.as_ptr().add(nx as usize).cast(),
        buf.as_ptr().add((nx+ny)as usize).cast(),warmup,samples,check as u32,times.as_mut_ptr(),&mut mismatches,&mut executions_checked,&mut diagnostics)})?;
    Ok(json!({"record":record["id"],"completed":true,"passed":mismatches==0,"mismatches":mismatches,"samples_ms":times,
        "executions_checked":executions_checked,"check":format!("{check:?}"),"diagnostics":diagnostics.json()}))
}

fn replay_segment(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32,check:ReplayCheck)->Result<Value,String>{
    if number(record,"version")?!=1 {return Err("unsupported segment corpus version".into());}
    let p=&record["profile"];
    let profile=SegmentProfile {kind:small(p,"kind")?,tokens:small(p,"tokens")?,batch:small(p,"batch")?,
        hidden:small(p,"hidden")?,intermediate:small(p,"intermediate")?,max_keys:small(p,"max_keys")?,
        slots:small(p,"slots")?,flags:small(p,"flags")?,rms_eps:p["rms_eps"].as_f64().ok_or("invalid segment epsilon")? as f32};
    if !(1..=3).contains(&profile.kind) || profile.tokens!=8 || profile.batch!=1 || profile.flags!=0 ||
        profile.hidden!=5120 || !profile.rms_eps.is_finite() || profile.rms_eps<=0.0 {return Err("unsupported segment profile".into());}
    let bindings=record["bindings"].as_array().ok_or("missing segment bindings")?;
    if bindings.is_empty() || bindings.len()>32 {return Err("invalid segment binding count".into());}
    let names:Vec<CString>=bindings.iter().map(|b|CString::new(b["name"].as_str().ok_or("missing segment binding name")?)
        .map_err(|e|e.to_string())).collect::<Result<_,String>>()?;
    let mut views=Vec::new();let mut pre=0u64;let mut post=0u64;
    for (binding,name) in bindings.iter().zip(names.iter()) {
        let flags=small(binding,"flags")?;let dtype=small(binding,"dtype")?;
        if flags==0 || flags>7 || dtype>3 {return Err("invalid segment binding flags/dtype".into());}
        let present=binding["present"].as_bool().ok_or("missing segment presence")?;
        let shape=binding["shape"].as_array().ok_or("missing segment shape")?;
        if shape.len()!=4 {return Err("invalid segment rank".into());}
        let mut dims=[0i32;4];let mut elements=1u64;
        for (i,n) in shape.iter().enumerate(){let n=n.as_u64().filter(|&v|v>0 && v<=1048576).ok_or("invalid segment extent")?;
            dims[i]=n as i32;elements=elements.checked_mul(n).ok_or("segment size overflow")?;}
        let bytes=if present {elements.checked_mul(match dtype {0=>2,1=>1,_=>4}).ok_or("segment size overflow")?} else {0};
        if flags&3!=0 {pre=pre.checked_add(bytes).ok_or("segment size overflow")?;}
        if flags&6!=0 {post=post.checked_add(bytes).ok_or("segment size overflow")?;}
        views.push(SegmentView{name:name.as_ptr(),flags,dtype,present:u32::from(present),shape:dims});
    }
    let bytes=pre.checked_add(post).ok_or("segment size overflow")?;
    if pre!=number(record,"pre_bytes")? || post!=number(record,"post_bytes")? || bytes!=number(record,"bytes")? || bytes>1<<31 {
        return Err("invalid segment byte counts".into());
    }
    let offset=number(record,"offset")?;let mut file=std::fs::File::open(data).map_err(|e|e.to_string())?;
    if offset.checked_add(bytes).ok_or("segment offset overflow")?>file.metadata().map_err(|e|e.to_string())?.len(){return Err("truncated segment trace".into());}
    file.seek(SeekFrom::Start(offset)).map_err(|e|e.to_string())?;
    let mut blob=vec![0u8;bytes as usize];file.read_exact(&mut blob).map_err(|e|e.to_string())?;
    let mut times=vec![0f32;samples as usize];let mut executions_checked=0;let mut diagnostics=Diagnostics::default();
    let mut per_binding=vec![Diagnostics::default();bindings.len()];let mut state_mismatches=0;let mut state_bytes=0;
    let mut binding_state_mismatches=vec![0u64;bindings.len()];let mut binding_state_bytes=vec![0u64;bindings.len()];
    checked(unsafe{ignis_lab_segment_replay(model.handle(),small(record,"layer")?,&profile,views.as_ptr(),views.len() as u32,
        blob.as_ptr().cast(),bytes,warmup,samples,check as u32,times.as_mut_ptr(),&mut executions_checked,&mut diagnostics,
        per_binding.as_mut_ptr(),&mut state_mismatches,&mut state_bytes,binding_state_mismatches.as_mut_ptr(),binding_state_bytes.as_mut_ptr())})?;
    let implementation=unsafe{CStr::from_ptr(ignis_lab_segment_implementation(model.handle()))}.to_string_lossy().into_owned();
    let outputs:Vec<Value>=bindings.iter().zip(per_binding.iter()).filter(|(b,_)|b["flags"].as_u64().unwrap()&4!=0)
        .map(|(b,d)|json!({"binding":b["name"],"diagnostics":d.json()})).collect();
    let preservation:Vec<Value>=bindings.iter().enumerate().filter(|(i,_)|binding_state_bytes[*i]>0).map(|(i,b)|
        json!({"binding":b["name"],"mismatched_bytes":binding_state_mismatches[i],"checked_bytes":binding_state_bytes[i],
            "readonly":b["flags"].as_u64().unwrap()&1!=0,"exact_match":binding_state_mismatches[i]==0})).collect();
    Ok(json!({"record":record["id"],"kind":"segment","segment_kind":profile.kind,"completed":true,
        "passed":diagnostics.mismatches==0 && state_mismatches==0,"mismatches":diagnostics.mismatches,
        "samples_ms":times,"executions_checked":executions_checked,"check":format!("{check:?}"),
        "diagnostics":diagnostics.json(),"outputs":outputs,"implementation":implementation,
        "candidate_applied":implementation!="builtin","state_diagnostics":{"mismatched_bytes":state_mismatches,
            "checked_bytes":state_bytes,"exact_match":state_mismatches==0,"bindings":preservation,
            "scope":"readonly boundary preservation, mutable nonnumeric state and GDN invalid record suffix preservation"},
        "timing_scope":"complete finite segment; isolated preimage/workspace reset excluded"}))
}

// Validate the disk envelope before allocating or passing pointers through FFI.
// Native validation additionally checks exact geometry and all device metadata.
fn gqa_view(record:&Value)->Result<(GqaView,u64),String>{
    if number(record,"version")?!=1 || number(record,"tokens")?!=8 {return Err("unsupported GQA recording version/width".into());}
    let mut view=GqaView {version:1,layer:small(record,"layer")?,max_keys:small(record,"max_keys")?,
        scale:record["scale"].as_f64().ok_or("missing GQA scale")? as f32,tensors:[GqaTensor::default();15]};
    let tensors=record["tensors"].as_array().ok_or("missing GQA tensors")?;
    if tensors.len()!=15 {return Err("invalid GQA tensor inventory".into());}
    let mut pre=0u64;let mut state=0u64;let mut output=0u64;
    for (i,tensor) in tensors.iter().enumerate(){
        let dtype=small(tensor,"dtype")?;
        let element_bytes=match dtype {0=>2,1=>1,2=>4,_=>return Err("invalid GQA dtype".into())};
        let shape=tensor["shape"].as_array().ok_or("invalid GQA shape")?;
        if shape.len()!=4 {return Err("invalid GQA rank".into());}
        let mut elements=1u64;
        for (dim,value) in shape.iter().enumerate(){
            let n=value.as_u64().filter(|&v|v<=1048576).ok_or("invalid GQA extent")?;
            view.tensors[i].shape[dim]=n as i32;
            elements=elements.checked_mul(n).ok_or("GQA size overflow")?;
        }
        if elements==0 && (i!=5 || view.tensors[i].shape!=[0;4]) {return Err("invalid empty GQA tensor".into());}
        view.tensors[i].dtype=dtype;
        let bytes=elements.checked_mul(element_bytes).ok_or("GQA size overflow")?;
        pre=pre.checked_add(bytes).ok_or("GQA size overflow")?;
        if i>=8 {state=state.checked_add(bytes).ok_or("GQA size overflow")?;}
        if i==0 {output=bytes;}
    }
    let bytes=pre.checked_add(output).and_then(|v|v.checked_add(state)).ok_or("GQA size overflow")?;
    if pre!=number(record,"pre_bytes")? || output!=number(record,"output_bytes")? ||
        state!=number(record,"state_bytes")? || bytes!=number(record,"bytes")? || bytes>1<<31 {
        return Err("invalid GQA sample byte counts".into());
    }
    Ok((view,bytes))
}
fn replay_gqa(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32,check:ReplayCheck)->Result<Value,String>{
    let (view,bytes)=gqa_view(record)?;let offset=number(record,"offset")?;
    let mut file=std::fs::File::open(data).map_err(|e|e.to_string())?;
    if offset.checked_add(bytes).ok_or("GQA offset overflow")?>file.metadata().map_err(|e|e.to_string())?.len(){
        return Err("truncated GQA trace data".into());
    }
    file.seek(SeekFrom::Start(offset)).map_err(|e|e.to_string())?;
    let mut blob=vec![0u8;bytes as usize];file.read_exact(&mut blob).map_err(|e|e.to_string())?;
    let mut times=vec![0f32;samples as usize];let mut executions_checked=0;
    let mut diagnostics=Diagnostics::default();let mut state_mismatches=0;let mut state_bytes=0;
    checked(unsafe{ignis_lab_gqa_replay(model.handle(),&view,blob.as_ptr().cast(),bytes,warmup,samples,
        check as u32,times.as_mut_ptr(),&mut executions_checked,&mut diagnostics,&mut state_mismatches,&mut state_bytes)})?;
    Ok(json!({"record":record["id"],"kind":"gqa_attention","completed":true,
        "passed":diagnostics.mismatches==0 && state_mismatches==0,"mismatches":diagnostics.mismatches,
        "samples_ms":times,"executions_checked":executions_checked,"check":format!("{check:?}"),
        "diagnostics":diagnostics.json(),"state_diagnostics":{"mismatched_bytes":state_mismatches,
        "checked_bytes":state_bytes,"exact_match":state_mismatches==0,
        "scope":"complete physical HQ code/meta, BF16 residual K/V and shared ring; all pool slots"},
        "timing_scope":"attention graph only; isolated input/state restore and diagnostics excluded",
        "implementation":gqa_status(model)["implementation"],"candidate_applied":gqa_status(model)["extension_loaded"]}))
}

/// Availability metadata; actual application is reported by each replay row.
pub fn segment_status(model:&Model)->Value {
    let implementation=unsafe{CStr::from_ptr(ignis_lab_segment_implementation(model.handle()))}.to_string_lossy().into_owned();
    json!({"extension_loaded":implementation!="builtin","implementation":implementation,
        "supported_scopes":["MLP flattened T8","GQA C1/T8/HQ","GDN verify C1/T8"],"unsupported_shapes":"builtin",
        "workspace_bytes":(1..=3).map(|kind|unsafe{ignis_lab_segment_workspace_capacity(model.handle(),kind)}).collect::<Vec<_>>()})
}

/// Active native attention entry for the supported C1/T8/HQ profile.
/// Other shapes/formats explicitly continue through the builtin dispatcher.
pub fn gqa_status(model:&Model)->Value {
    let implementation=unsafe{CStr::from_ptr(ignis_lab_gqa_implementation(model.handle()))}.to_string_lossy().into_owned();
    json!({"implementation":implementation,"extension_loaded":unsafe{ignis_lab_gqa_candidate_applied(model.handle())}!=0,
        "supported_scope":"C1/T8/HQ","unsupported_shapes":"builtin"})
}

pub fn minimum(model:&Model,names:&str,tokens:u64,requests:u64)->Result<u64,String>{
    minimum_filtered(model,names,tokens,requests,&[])
}
/// A conservative prefill lower bound: filtered prompt chunk sizes are unknown,
/// so only the selected one-row head is guaranteed to contribute.
pub fn minimum_filtered(model:&Model,names:&str,tokens:u64,requests:u64,token_filter:&[u32])->Result<u64,String>{
    let count=u32::try_from(token_filter.len()).map_err(|e|e.to_string())?;
    let names=CString::new(names).map_err(|e|e.to_string())?;let mut bytes=0;
    checked(unsafe{ignis_lab_capture_minimum_filtered(model.handle(),names.as_ptr(),tokens,requests,
        token_filter.as_ptr(),count,&mut bytes)})?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gqa_record()->Value {
        let shapes=[[256,24,8,1],[256,4,8,1],[256,4,8,1],[256,24,8,1],
            [8,1,1,1],[1,1,1,1],[1,1,1,1],[16,2,1,1],
            [64,64,4,32],[64,64,4,32],[8,64,4,32],[8,64,4,32],
            [256,4,544,2],[256,4,544,2],[16,2,1,1]];
        let mut pre=0u64;let mut state=0u64;let mut output=0u64;
        let tensors:Vec<Value>=shapes.iter().enumerate().map(|(i,shape)|{
            let dtype=if i<4 || i==12 || i==13 {0} else if (8..=11).contains(&i) {1} else {2};
            let bytes=shape.iter().map(|&v|v as u64).product::<u64>()*match dtype {0=>2,1=>1,_=>4};
            pre+=bytes;if i>=8 {state+=bytes;}if i==0 {output=bytes;}
            json!({"dtype":dtype,"shape":shape})
        }).collect();
        json!({"version":1,"layer":3,"tokens":8,"max_keys":1024,"scale":0.0625,
            "pre_bytes":pre,"output_bytes":output,"state_bytes":state,"bytes":pre+output+state,"tensors":tensors})
    }
    #[test]
    fn gqa_disk_envelope_rejects_version_counts_extents_and_overflow() {
        let valid=gqa_record();assert!(gqa_view(&valid).is_ok());
        for field in ["version","tokens","pre_bytes","output_bytes","state_bytes","bytes"] {
            let mut damaged=valid.clone();damaged[field]=json!(2);
            assert!(gqa_view(&damaged).is_err(),"accepted corrupt {field}");
        }
        let mut damaged=valid.clone();damaged["tensors"][0]["shape"]=json!([1048576,1048576,1048576,1048576]);
        assert!(gqa_view(&damaged).is_err());
        damaged=valid.clone();damaged["tensors"][0]["shape"]=json!([-1,24,8,1]);
        assert!(gqa_view(&damaged).is_err());
        damaged=valid.clone();damaged["tensors"][0]["shape"]=json!([256,24,8]);
        assert!(gqa_view(&damaged).is_err());
        damaged=valid.clone();damaged["tensors"][1]["dtype"]=json!(99);
        assert!(gqa_view(&damaged).is_err());
        damaged=valid;damaged["tensors"].as_array_mut().unwrap().pop();
        assert!(gqa_view(&damaged).is_err());
    }
    #[test]
    fn diagnostics_distinguish_zero_reference_and_missing_finite_pairs() {
        let mut d=Diagnostics::default();
        assert!(d.json()["relative_l2"].is_null());
        assert!(d.json()["max_abs_error"].is_null());
        d.finite_elements=4;
        assert_eq!(d.json()["relative_l2"],0.0);
        d.squared_error=4.0;
        assert!(d.json()["relative_l2"].is_null());
        assert_eq!(d.json()["relative_l2_zero_reference"],true);
        d.squared_reference=16.0;
        assert_eq!(d.json()["relative_l2"],0.5);
        assert_eq!(d.json()["relative_l2_zero_reference"],false);
        d.actual_nan=2;d.reference_inf=3;
        assert_eq!(d.json()["actual_nan"],2);
        assert_eq!(d.json()["reference_inf"],3);
    }
}
