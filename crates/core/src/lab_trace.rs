//! Experimental persistent Trellis samples. No GPU addresses cross the file boundary.
#![cfg(feature = "cuda")]
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use serde_json::{json, Value};
use crate::model_load::{Model, ffi::IgnisModel};
use crate::seq::{SeqPool, ffi::IgnisSeqPool};

#[repr(C)]
struct View {
    name:*const c_char, trellis_offset:u64, suh_offset:u64, svh_offset:u64,
    k2:u32, input:u32, output:u32, tile_stride:u32,
}
unsafe extern "C" {
    fn ignis_lab_capture_start(m:*mut IgnisModel,p:*mut IgnisSeqPool,dir:*const c_char,pin:u64,disk:u64,names:*const c_char)->i32;
    fn ignis_lab_capture_stop(m:*mut IgnisModel,p:*mut IgnisSeqPool)->i32;
    fn ignis_lab_trace_context(m:*mut IgnisModel,request:u32,phase:u32,position:u64)->i32;
    fn ignis_lab_trace_flush(m:*mut IgnisModel)->i32;
    fn ignis_lab_capture_minimum(m:*mut IgnisModel,names:*const c_char,tokens:u64,requests:u64,bytes:*mut u64)->i32;
    fn ignis_lab_trace_error()->*const c_char;
    fn ignis_lab_replay(m:*mut IgnisModel,w:*const View,up:*const View,t:u32,ldy:u32,epilogue:u32,
        x:*const c_void,initial:*const c_void,reference:*const c_void,warmup:u32,samples:u32,check:u32,
        times:*mut f32,mismatches:*mut u64,checked:*mut u32)->i32;
}
fn checked(rc:i32)->Result<(),String>{
    if rc==0 {Ok(())} else {Err(unsafe{CStr::from_ptr(ignis_lab_trace_error())}.to_string_lossy().into_owned())}
}
pub fn start(model:&mut Model,pool:&mut SeqPool,path:&Path,pin:u64,disk:u64,names:&str)->Result<(),String>{
    let dir=CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e|e.to_string())?;
    let names=CString::new(names).map_err(|e|e.to_string())?;
    checked(unsafe{ignis_lab_capture_start(model.handle(),pool.handle(),dir.as_ptr(),pin,disk,names.as_ptr())})
}
pub fn stop(model:&mut Model,pool:&mut SeqPool)->Result<(),String>{
    checked(unsafe{ignis_lab_capture_stop(model.handle(),pool.handle())})
}
pub fn context(model:&Model,request:u32,phase:u32,position:u64)->Result<(),String>{
    checked(unsafe{ignis_lab_trace_context(model.handle(),request,phase,position)})
}
pub fn flush(model:&Model)->Result<(),String>{checked(unsafe{ignis_lab_trace_flush(model.handle())})}
fn number(v:&Value,key:&str)->Result<u64,String>{v[key].as_u64().ok_or_else(||format!("invalid trace field {key}"))}
fn small(v:&Value,key:&str)->Result<u32,String>{u32::try_from(number(v,key)?).map_err(|e|e.to_string())}
fn view(v:&Value,name:&CString)->Result<View,String>{Ok(View{name:name.as_ptr(),
    trellis_offset:number(v,"trellis_offset")?,suh_offset:number(v,"suh_offset")?,svh_offset:number(v,"svh_offset")?,
    k2:small(v,"k2")?,input:small(v,"in")?,output:small(v,"out")?,tile_stride:small(v,"tile_stride")?})}

/// Regression callers check every execution, including warmups.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum ReplayCheck { FirstExecution = 0, EveryExecution = 1 }

pub fn replay(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32)->Result<Value,String>{
    replay_with_check(model,data,record,warmup,samples,ReplayCheck::EveryExecution)
}

pub fn replay_with_check(model:&mut Model,data:&Path,record:&Value,warmup:u32,samples:u32,check:ReplayCheck)->Result<Value,String>{
    if samples==0 || samples>10000 || warmup>10000 {return Err("invalid sample counts".into());}
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
    checked(unsafe{ignis_lab_replay(model.handle(),&w,up.as_ref().map_or(std::ptr::null(),|v|v),t,ldy,
        small(record,"epilogue")?,buf.as_ptr().cast(),buf.as_ptr().add(nx as usize).cast(),
        buf.as_ptr().add((nx+ny)as usize).cast(),warmup,samples,check as u32,times.as_mut_ptr(),&mut mismatches,&mut executions_checked)})?;
    Ok(json!({"record":record["id"],"passed":mismatches==0,"mismatches":mismatches,"samples_ms":times,"executions_checked":executions_checked,"check":format!("{check:?}")}))
}

pub fn minimum(model:&Model,names:&str,tokens:u64,requests:u64)->Result<u64,String>{
    let names=CString::new(names).map_err(|e|e.to_string())?;let mut bytes=0;
    checked(unsafe{ignis_lab_capture_minimum(model.handle(),names.as_ptr(),tokens,requests,&mut bytes)})?;
    Ok(bytes)
}
