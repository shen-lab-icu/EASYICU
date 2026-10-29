"""Trusted CSV bridge with a handwritten multi-export DOM, never artifact JS."""

from tests.webserver.copilot.test_pi_workspace_csv_filter import _run


def test_declared_and_semantic_multi_exports_require_explicit_choice_and_preserve_filters():
    _run(r"""
(async()=>{
  for(const noContract of [false,true]){
    const page=setup({multi:true,noContract});page.change('F','SICU');page.allow();await page.settle();
    assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
    assert.deepEqual(page.slot.children.slice(1).map(node=>node.textContent),['下载当前筛选 CSV','下载原始 CSV']);
    const manifest=page.messages.find(data=>data.type==='easyicu:workspace-csv-options');
    assert.equal(manifest.options.length,2);
    assert.ok(manifest.options.every(option=>/^[a-f0-9]{32}$/.test(option.token)));
    assert.deepEqual(Object.keys(manifest.options[0]).sort(),['label','token']);
    page.pick(0);await page.settle();assert.equal(page.csv().trim().split('\n').length,6);
    assert.deepEqual(page.filters,{sex:'F',unit:'SICU'});
    page.generate();await page.settle();page.pick(1);await page.settle();
    assert.equal(page.csv().trim().split('\n').length,25);
    assert.equal(page.slot.children[0].download,'original.csv');
    assert.deepEqual(page.filters,{sex:'F',unit:'SICU'});assert.equal(page.nativeClicks(),0);
    page.cleanup();
  }
  console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_multi_export_choice_is_bounded_to_eight_and_never_auto_selects_nine():
    _run(r"""
(async()=>{
  const eight=setup({count:8});eight.allow();await eight.settle();
  assert.equal(eight.slot.children.length,9);assert.equal(eight.clicks(),0);
  eight.pick(7);await eight.settle();assert.equal(eight.clicks(),1);eight.cleanup();
  const nine=setup({count:9});nine.allow();await nine.settle();
  assert.equal(nine.slot.children.length,1);assert.equal(nine.slot.children[0].tag,'span');
  assert.equal(nine.clicks(),0);assert.equal(nine.calls.length,0);nine.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_forged_selection_tokens_sources_nonces_epochs_and_versions_cannot_click():
    _run(r"""
(async()=>{
  const page=setup({multi:true});page.allow();await page.settle();
  const options=page.messages.find(data=>data.type==='easyicu:workspace-csv-options');
  const data={type:'easyicu:workspace-csv-control',nonce:page.nonce,epoch:options.epoch,
    selection_version:options.selection_version,action:'select',token:options.options[0].token};
  for(const invalid of [
    {...data,nonce:'e'.repeat(64)},{...data,epoch:'f'.repeat(64)},
    {...data,selection_version:99},{...data,token:'e'.repeat(32)},
    {...data,token:'#downloadOriginal'},{...data,token:'javascript:alert(1)'},
  ]){
    page.wrapper({data:invalid,source:page.hostProxy,origin:'http://localhost'});
    page.child({data:invalid,source:page.wrapperChildProxy,origin:'null'});
  }
  page.wrapper({data,source:{},origin:'http://localhost'});
  page.child({data,source:{},origin:'null'});page.child({data,source:page.wrapperChildProxy,origin:'http://localhost'});
  await page.settle();assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  // Unknown selector/script fields never cross the wrapper; only its current token does.
  page.wrapper({data:{...data,selector:'#downloadOriginal',script:'throw new Error("unsafe")'},source:page.hostProxy,origin:'http://localhost'});
  await page.settle();assert.equal(page.clicks(),1);assert.equal(page.slot.children[0].download,'synthetic.csv');
  page.wrapper({data,source:page.hostProxy,origin:'http://localhost'});
  page.child({data,source:page.wrapperChildProxy,origin:'null'});
  await page.settle();assert.equal(page.clicks(),1);page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_old_parent_buttons_are_inert_after_selection_epoch_hash_close_or_frame_changes():
    _run(r"""
(async()=>{
  for(const change of [page=>page.change('F','SICU'),page=>page.generate(),
    page=>page.changedHash(),page=>page.revoke(),page=>{page.frame.isConnected=false;}]){
    const page=setup({multi:true});page.allow();await page.settle();
    const button=page.slot.children[1];change(page);
    button.handlers.click({preventDefault(){},stopPropagation(){}});
    await page.settle();assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);page.cleanup();
  }
  console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_selected_element_must_still_be_enabled_connected_and_in_the_current_allowlist():
    _run(r"""
(async()=>{
  for(const change of [page=>{page.controls[0].disabled=true;},page=>{page.controls[0].isConnected=false;},
    page=>{page.controls[0]={...page.controls[0]};},page=>{page.filters.sex='F';}]){
    const page=setup({multi:true});page.allow();await page.settle();change(page);page.pick(0);
    await page.settle();assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);page.cleanup();
  }
  console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_duplicate_options_late_ack_and_repeated_button_click_cannot_repeat_or_replace_an_export():
    _run(r"""
(async()=>{
  const page=setup({multi:true});page.allow();await page.settle();
  const options=page.messages.find(data=>data.type==='easyicu:workspace-csv-options'),button=page.slot.children[1];
  const childOptions={...options,type:'easyicu:workspace-csv-child-options'};
  page.wrapper({data:childOptions,source:page.childProxy,origin:'null'});
  await page.host({data:options,source:page.wrapperProxy,origin:'null'});
  await page.host({data:{type:'easyicu:workspace-csv-ack',nonce:page.nonce,epoch:options.epoch,status:'manual'},source:page.wrapperProxy,origin:'null'});
  assert.equal(page.slot.children[1],button);
  button.handlers.click({preventDefault(){},stopPropagation(){}});
  button.handlers.click({preventDefault(){},stopPropagation(){}});
  await page.settle();assert.equal(page.clicks(),1);assert.equal(page.calls.length,1);
  const link=page.slot.children[0];
  await page.host({data:options,source:page.wrapperProxy,origin:'null'});
  page.wrapper({data:childOptions,source:page.childProxy,origin:'null'});
  await page.settle();assert.equal(page.slot.children[0],link);
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")


def test_option_labels_are_inert_text_and_qa_active_selection_fails_closed():
    _run(r"""
(async()=>{
  const label='下载 CSV <img src=x onerror=alert(1)>',page=setup({multi:true,labels:[label,'下载原始 CSV']});
  page.allow();await page.settle();assert.equal(page.slot.children[1].textContent,label);
  assert.equal(page.slot.children[1].children.length,0);
  page.childContext.__easyicuRuntimeQaRunning=true;page.pick(0);await page.settle();
  assert.equal(page.clicks(),0);assert.equal(page.calls.length,0);
  assert.ok(page.slot.children[0].textContent.includes('checks are still running'));
  page.cleanup();console.log('passed');
})().catch(error=>{console.error(error);process.exit(1);});
""")
