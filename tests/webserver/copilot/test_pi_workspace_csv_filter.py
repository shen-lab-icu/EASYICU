"""The passive nested-frame CSV bridge preserves live filter state and consent."""

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.workspace_csv_download import csv_preview_bridge
from tests.support.node import run_node
from tests.support.workspace_csv import _run




def test_opt_in_generates_five_current_rows_without_navigation_or_native_download() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  assert.ok(page.allow());await page.settle();
  assert.deepEqual(page.filters,{sex:'F',unit:'SICU'});
  assert.equal(page.clicks(),1);assert.equal(page.calls.length,1);
  assert.equal(page.csv().trim().split('\n').length,6);
  assert.equal(page.slot.children[0].tag,'a');assert.ok(page.slot.children[0].textContent.includes('click to save'));
  assert.equal(page.nativeClicks(),0);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_passive_preview_cannot_forward_or_save_a_csv_before_explicit_opt_in() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');page.manualExport();await page.settle();
  assert.equal(page.nativeClicks(),0);assert.equal(page.calls.length,0);
  assert.equal(page.messages.filter(data=>data.type==='easyicu:workspace-csv').length,0);
  const data={type:'easyicu:workspace-csv-child',nonce:page.nonce,epoch:'',selection_version:1,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n1\n'};
  page.wrapper({data,source:page.childProxy,origin:'null'});await page.settle();
  assert.equal(page.calls.length,0);assert.equal(page.slot.children.length,0);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_filter_change_invalidates_old_save_link_and_new_generation_has_fresh_epoch() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.allow();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,25);
  const oldLink=page.slot.children[0],oldEpoch=page.commands.at(-1).epoch;
  page.change('F','SICU');await page.settle();
  assert.ok(page.revoked.includes(oldLink.href));assert.equal(page.slot.children[0].tag,'span');
  let prevented=false;oldLink.handlers.click({preventDefault(){prevented=true;}});assert.ok(prevented);
  assert.ok(page.generate());assert.notEqual(page.commands.at(-1).epoch,oldEpoch);await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  assert.equal(page.slot.children[0].tag,'a');page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_sources_nonces_old_epochs_repeated_commands_and_acks_fail_closed() -> None:
    _run(r"""
(async()=>{
  const page=setup(),control={type:'easyicu:workspace-csv-control',nonce:page.nonce,action:'generate',epoch:'d'.repeat(64)};
  page.wrapper({data:control,source:{},origin:'http://localhost'});
  page.wrapper({data:{...control,nonce:'e'.repeat(64)},source:page.hostProxy,origin:'http://localhost'});
  page.child({data:control,source:{},origin:'null'});
  page.child({data:control,source:page.wrapperChildProxy,origin:'http://localhost'});
  await page.settle();assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  const csv={type:'easyicu:workspace-csv',nonce:page.nonce,epoch:'d'.repeat(64),selection_version:0,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n1\n'};
  await page.host({data:csv,source:page.wrapperProxy,origin:'null'});assert.equal(page.calls.length,0);
  page.allow();const epoch=page.commands.at(-1).epoch;await page.settle();
  const calls=page.calls.length,clicks=page.clicks(),link=page.slot.children[0];
  page.wrapper({data:{...control,epoch},source:page.hostProxy,origin:'http://localhost'});
  await page.host({data:{type:'easyicu:workspace-csv-ack',nonce:page.nonce,epoch,status:'manual'},source:{},origin:'null'});
  await page.host({data:{type:'easyicu:workspace-csv-ack',nonce:page.nonce,epoch,status:'manual'},source:page.wrapperProxy,origin:'null'});
  await page.settle();assert.equal(page.clicks(),clicks);assert.equal(page.slot.children[0],link);
  page.revoke();await page.host({data:{...csv,epoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);assert.equal(page.slot.children.length,0);
  page.allow();const newEpoch=page.commands.at(-1).epoch;assert.notEqual(epoch,newEpoch);
  await page.host({data:{...csv,epoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);page.changedHash();
  await page.host({data:{...csv,epoch:newEpoch},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,calls);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_qa_active_export_does_not_click_temporary_filters() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.childContext.__easyicuRuntimeQaRunning=true;
  page.allow();await page.settle();
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  assert.ok(page.slot.children[0].textContent.includes('checks are still running'));
  page.childContext.__easyicuRuntimeQaRunning=false;page.change('F','SICU');
  page.generate();await page.settle();assert.equal(page.csv().trim().split('\n').length,6);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_initial_loading_queues_generation_and_repeated_ready_does_not_repeat_export() -> None:
    _run(r"""
(async()=>{
  const page=setup({loading:true});page.change('F','SICU');page.allow();await page.settle();
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  const ready={data:{type:'easyicu:workspace-csv-ready-control',nonce:page.nonce},source:page.wrapperProxy,origin:'null'};
  await page.host({...ready,source:{}});await page.host({...ready,data:{...ready.data,nonce:'e'.repeat(64)}});
  await page.host(ready);await page.host(ready);page.loaded();await page.settle();
  assert.equal(page.clicks(),1);assert.equal(page.calls.length,1);
  assert.equal(page.csv().trim().split('\n').length,6);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_filter_change_and_disable_abort_inflight_host_reply_and_old_payloads() -> None:
    _run(r"""
(async()=>{
  const page=setup(),original=page.hostContext.fetch;
  let finish,pendingSignal;
  page.hostContext.fetch=(url,options)=>{pendingSignal=options.signal;return new Promise(resolve=>{
    finish=async()=>resolve(await original(url,options));
  });};
  page.allow();
  // Wait for the child's Blob promise and Host request without awaiting its deferred response.
  await new Promise(resolve=>setImmediate(resolve));
  const epoch=page.commands.at(-1).epoch;
  // Generation still needs its two animation frames.
  const settling=page.settle();await new Promise(resolve=>setImmediate(resolve));
  assert.ok(finish);page.change('F','SICU');assert.ok(pendingSignal.aborted);
  await finish();await settling;assert.equal(page.slot.children[0].tag,'span');
  const previousCalls=page.calls.length;
  const payload={type:'easyicu:workspace-csv',nonce:page.nonce,epoch,selection_version:0,
    filename:'synthetic.csv',media_type:'text/csv',csv:'id\n24\n'};
  await page.host({data:payload,source:page.wrapperProxy,origin:'null'});
  assert.equal(page.calls.length,previousCalls);
  page.hostContext.fetch=original;page.generate();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  page.revoke();assert.equal(page.slot.children.length,0);
  await page.host({data:{...payload,epoch:page.commands.at(-2).epoch,selection_version:1},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.slot.children.length,0);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_real_programmatic_reset_invalidates_ready_csv_at_first_microtask_without_raw_values() -> None:
    _run(r"""
(async()=>{
  const page=setup();page.change('F','SICU');page.allow();await page.settle();
  assert.equal(page.csv().trim().split('\n').length,6);
  const link=page.slot.children[0];
  page.clickWithHandler(()=>page.manualExport());await Promise.resolve();
  assert.equal(page.slot.children[0],link);await page.settle();
  assert.equal(page.slot.children[0],link,'Export click with unchanged filters must preserve its ready link');
  page.resetPage();await Promise.resolve();
  assert.deepEqual(page.filters,{sex:'all',unit:'all'});
  assert.ok(page.revoked.includes(link.href),'A synchronous reset must revoke the old link before the next user task');
  assert.equal(page.slot.children[0].tag,'span');
  const changes=page.messages.filter(data=>data.type==='easyicu:workspace-csv-changed');
  assert.equal(changes.length,1);
  assert.deepEqual(Object.keys(changes[0]).sort(),['epoch','nonce','selection_version','type']);
  await page.settle();assert.equal(page.messages.filter(data=>data.type==='easyicu:workspace-csv-changed').length,1);
  page.generate();await page.settle();assert.equal(page.csv().trim().split('\n').length,25);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")
