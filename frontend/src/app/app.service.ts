import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import { API } from 'enveriment';


@Injectable({
  providedIn: 'root'
})
export class AppService {

  constructor(private http: HttpClient) { }
  register(name:any,fname:any,oname:any,email:any,pas:any){
    let body = new FormData();
    body.append('name', name);
    body.append('surname', fname);
    body.append('patronymic', oname);
    body.append('login', email);
    body.append('password', pas);

    const req = this.http.post(API+'sign_up',body);
    return req
  }
  auth(email:any,pas:any){
    const req = this.http.get(API+'log_in?login='+email+'&password='+pas);
    return req
  }
  onCange(text:any){
    const req = this.http.get(API+'onchange?text='+text);
    return req
  }
  getCode(text:any){
    const req = this.http.get(API+'main?text='+text);
    return req
  }
  getlist(){
    const req = this.http.get(API+'getlist');
    return req
  }
  setlist(name:any,autor:any,code:any,date:any){
    const req = this.http.get(API+'setlist?name='+name+'&autor='+autor+'&code='+code+'&date='+date);
    return req
  }
}
